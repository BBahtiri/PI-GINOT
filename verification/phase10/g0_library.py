#!/usr/bin/env python3
"""
Phase 10, gate G0 -- the reference side.  Runs in the fastvpinns venv
(Python 3.10, fastvpinns 1.0.2, tensorflow-cpu 2.13.0), NOT in the project env:

    /home/claude/venvs/fv310/bin/python verification/phase10/g0_library.py tensors OUT.npz
    /home/claude/venvs/fv310/bin/python verification/phase10/g0_library.py train  OUT.json

`tensors`: builds the library's finite-element space on two quad meshes -- the
library's own internal 4x4 mesh of the unit square (rectangles, constant
Jacobian) and a randomly distorted 6x6 mesh written as a medit .mesh file
(general quadrilaterals, Jacobian varying inside the cell) -- and saves the cell
vertices as the library orders them, the quadrature points, the pre-multiplied
test-function tensors and forcing matrix, plus a fixed float64 network's
weights, its Poisson residual matrix and loss.  The PyTorch port must reproduce
all of it (G0b).

`train`: the library trains its Poisson benchmark u = -sin(2 pi x) sin(2 pi y)
on the unit square (2x2 cells, 15 test functions per direction, 40x40
Gauss-Legendre points, 3x30 tanh, Adam 1e-3, beta 10, float32, 20,000 steps) and
reports the relative L2 error on a 100x100 grid (G0a, and the reference for G0c).
"""
import json
import os
import sys
import tempfile
import time

import meshio
import numpy as np
import tensorflow as tf

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model import DenseModel
from fastvpinns.physics.poisson2d import pde_loss_poisson

W = 2.0 * np.pi
exact = lambda x, y: -np.sin(W * x) * np.sin(W * y)
rhs = lambda x, y: -2.0 * W ** 2 * np.sin(W * x) * np.sin(W * y)   # -Lap u = f


def distorted_mesh(path, n=6, seed=3):
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X, Y], -1).reshape(-1, 2)
    inner = (P[:, 0] > 0) & (P[:, 0] < 1) & (P[:, 1] > 0) & (P[:, 1] < 1)
    P[inner] += rng.uniform(-0.25, 0.25, (inner.sum(), 2)) / n
    idx = lambda i, j: i * (n + 1) + j
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(n) for j in range(n)])
    lines, refs = [], []
    for k in range(n):
        lines += [[idx(k, 0), idx(k + 1, 0)], [idx(n, k), idx(n, k + 1)],
                  [idx(k + 1, n), idx(k, n)], [idx(0, k + 1), idx(0, k)]]
        refs += [1000, 1001, 1002, 1003]
    pts = np.hstack([P, np.zeros((len(P), 1))])
    meshio.write(path, meshio.Mesh(pts, [("quad", quads), ("line", np.array(lines))],
                                   cell_data={"medit:ref": [np.zeros(len(quads), int),
                                                            np.array(refs)]}))


def space(kind, fe_order, quad_order, out, n_bd=400):
    dom = Geometry_2D("quadrilateral", "internal" if kind == "internal" else "external",
                      50, 50, out)
    if kind == "internal":
        n = 4 if fe_order < 10 else 2
        cells, bpts = dom.generate_quad_mesh_internal(x_limits=[0, 1], y_limits=[0, 1],
                                                      n_cells_x=n, n_cells_y=n,
                                                      num_boundary_points=n_bd)
    else:
        f = os.path.join(out, "distorted.mesh")
        distorted_mesh(f)
        cells, bpts = dom.read_mesh(f, 3, "uniform", refinement_level=0)
    tags = list(bpts.keys())
    fs = Fespace2D(mesh=dom.mesh, cells=cells, boundary_points=bpts, cell_type="quadrilateral",
                   fe_order=fe_order, fe_type="jacobi", quad_order=quad_order,
                   quad_type="gauss-legendre", fe_transformation_type="bilinear",
                   bound_function_dict={t: (lambda x, y: exact(x, y)) for t in tags},
                   bound_condition_dict={t: "dirichlet" for t in tags},
                   forcing_function=rhs, output_path=out, generate_mesh_plot=False)
    return dom, cells, fs


def model_for(dh, fs, dtype, lr=1e-3, seed=0):
    tf.keras.utils.set_random_seed(seed)
    din, dact = dh.get_dirichlet_input()
    return DenseModel(layer_dims=[2, 30, 30, 30, 1],
                      learning_rate_dict={"initial_learning_rate": lr, "use_lr_scheduler": False,
                                          "decay_steps": 1000, "decay_rate": 0.99,
                                          "staircase": True},
                      params_dict={"n_cells": fs.n_cells}, loss_function=pde_loss_poisson,
                      input_tensors_list=[dh.x_pde_list, din, dact],
                      orig_factor_matrices=[dh.shape_val_mat_list, dh.grad_x_mat_list,
                                            dh.grad_y_mat_list],
                      force_function_list=dh.forcing_function_list, tensor_dtype=dtype,
                      activation="tanh")


def residual(model, dh, fs):
    x = dh.x_pde_list
    with tf.GradientTape() as t:
        t.watch(x)
        u = model(x)
    g = t.gradient(u, x)
    nq = dh.grad_x_mat_list.shape[-1]
    gx = tf.reshape(g[:, 0], [fs.n_cells, nq])
    gy = tf.reshape(g[:, 1], [fs.n_cells, nq])
    rx = tf.transpose(tf.linalg.matvec(dh.grad_x_mat_list, gx))
    ry = tf.transpose(tf.linalg.matvec(dh.grad_y_mat_list, gy))
    R = rx + ry - dh.forcing_function_list                  # (N_test, N_cells), eps = 1
    cells = pde_loss_poisson(dh.shape_val_mat_list, dh.grad_x_mat_list, dh.grad_y_mat_list,
                             tf.reshape(u, [fs.n_cells, nq]), gx, gy,
                             dh.forcing_function_list, {"eps": tf.constant(1.0, tf.float64)})
    return R.numpy(), float(tf.reduce_sum(cells).numpy()), u.numpy(), g.numpy()


def tensors(out_npz):
    out = {}
    tmp = tempfile.mkdtemp()
    for kind, K, Q in (("internal", 5, 6), ("distorted", 4, 5)):
        dom, cells, fs = space(kind, K, Q, tmp)
        dh = DataHandler2D(fs, dom, tf.float64)
        m = model_for(dh, fs, tf.float64, seed=1)
        R, L, u, g = residual(m, dh, fs)
        p = kind + "_"
        out[p + "u"] = u
        out[p + "g"] = g
        out[p + "cells"] = np.asarray(cells, np.float64)
        out[p + "xq"] = dh.x_pde_list.numpy()
        out[p + "val"] = dh.shape_val_mat_list.numpy()
        out[p + "gx"] = dh.grad_x_mat_list.numpy()
        out[p + "gy"] = dh.grad_y_mat_list.numpy()
        out[p + "force"] = dh.forcing_function_list.numpy()
        out[p + "R"] = R
        out[p + "loss"] = np.array(L)
        out[p + "K"] = np.array(K)
        out[p + "Q"] = np.array(Q)
        for i, w in enumerate(m.get_weights()):
            out[p + f"w{i}"] = w
    np.savez(out_npz, **out)
    print("saved", out_npz, {k: v.shape for k, v in out.items() if k.endswith(("gx", "R"))})


def train(out_json, steps=20000):
    tmp = tempfile.mkdtemp()
    dom, cells, fs = space("internal", 15, 40, tmp, n_bd=800)
    dh = DataHandler2D(fs, dom, tf.float32)
    m = model_for(dh, fs, tf.float32, lr=1e-3, seed=0)
    bp = dh.get_bilinear_params_dict_as_tensors(lambda: {"eps": 1.0})
    t0 = time.time()
    hist = []
    for k in range(steps):
        l = m.train_step(beta=10, bilinear_params_dict=bp)
        if (k + 1) % 2000 == 0:
            hist.append((k + 1, float(l["loss"])))
    g = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], -1).astype(np.float32)
    u = m(tf.constant(P)).numpy().ravel()
    ue = exact(P[:, 0], P[:, 1])
    rel = float(np.linalg.norm(u - ue) / np.linalg.norm(ue))
    res = {"rel_L2": rel, "max_abs": float(np.abs(u - ue).max()), "steps": steps,
           "wall_s": time.time() - t0, "loss_hist": hist, "n_cells": fs.n_cells,
           "n_quad_per_cell": int(dh.grad_x_mat_list.shape[-1]),
           "n_test_per_cell": int(dh.grad_x_mat_list.shape[1])}
    json.dump(res, open(out_json, "w"), indent=1)
    print(res)


if __name__ == "__main__":
    {"tensors": tensors, "train": train}[sys.argv[1]](sys.argv[2])
