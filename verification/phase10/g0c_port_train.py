#!/usr/bin/env python3
"""
Phase 10, gate G0c -- the PyTorch port trains the library's Poisson benchmark.

Same problem and settings as verification/phase10/g0_library.py `train`:
u = -sin(2 pi x) sin(2 pi y) on the unit square, 2x2 cells, 15 test functions
per direction (bubbles), 40x40 Gauss-Legendre points, 3x30 tanh MLP with Glorot
init, Adam 1e-3, beta = 10 on 800 boundary points, float32, 20,000 steps; loss
sum_cells mean_tests R^2 + beta * mean (u - g)^2, exactly the library's.
Pass: relative L2 on a 100x100 grid within 2x of the library's own run.

    python -m verification.phase10.g0c_port_train
"""
import json
import time

import numpy as np
import torch

from physics.vpinn.tensors import quad_tensors

W = 2.0 * np.pi
exact = lambda x, y: -np.sin(W * x) * np.sin(W * y)
rhs = lambda x, y: -2.0 * W ** 2 * np.sin(W * x) * np.sin(W * y)
LIB = "verification/results/phase10/g0_library_train.json"
OUT = "verification/results/phase10/g0c.json"


def main(steps=20000, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    g = np.linspace(0, 1, 3)
    cells = np.array([[[g[i], g[j]], [g[i + 1], g[j]], [g[i + 1], g[j + 1]], [g[i], g[j + 1]]]
                      for i in range(2) for j in range(2)])
    T = quad_tensors(cells, 15, 40)
    dt = torch.float32
    x = torch.tensor(T.x, dtype=dt, requires_grad=True)
    gx, gy = torch.tensor(T.gx, dtype=dt), torch.tensor(T.gy, dtype=dt)
    F = torch.tensor(np.einsum("ekq,eq->ek", T.val, rhs(T.x[:, 0], T.x[:, 1]).reshape(T.n_elem, -1)),
                     dtype=dt)
    s = np.linspace(0, 1, 201)[:-1]
    bd = np.concatenate([np.stack([s, 0 * s], 1), np.stack([1 + 0 * s, s], 1),
                         np.stack([1 - s, 1 + 0 * s], 1), np.stack([0 * s, 1 - s], 1)])
    xb = torch.tensor(bd, dtype=dt)
    ub = torch.tensor(exact(bd[:, 0], bd[:, 1])[:, None], dtype=dt)
    layers = []
    for a, b in ((2, 30), (30, 30), (30, 30), (30, 1)):
        lin = torch.nn.Linear(a, b)
        torch.nn.init.xavier_uniform_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        layers += [lin, torch.nn.Tanh()]
    net = torch.nn.Sequential(*layers[:-1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0, hist = time.time(), []
    for k in range(steps):
        opt.zero_grad()
        u = net(x)
        du, = torch.autograd.grad(u.sum(), x, create_graph=True)
        ux = du[:, 0].reshape(T.n_elem, -1)
        uy = du[:, 1].reshape(T.n_elem, -1)
        R = torch.einsum("ekq,eq->ek", gx, ux) + torch.einsum("ekq,eq->ek", gy, uy) - F
        loss = (R ** 2).mean(1).sum() + 10.0 * ((net(xb) - ub) ** 2).mean()
        loss.backward()
        opt.step()
        if (k + 1) % 2000 == 0:
            hist.append((k + 1, float(loss)))
    gg = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(gg, gg)
    P = np.stack([X.ravel(), Y.ravel()], -1)
    with torch.no_grad():
        up = net(torch.tensor(P, dtype=dt)).numpy().ravel()
    ue = exact(P[:, 0], P[:, 1])
    rel = float(np.linalg.norm(up - ue) / np.linalg.norm(ue))
    lib = json.load(open(LIB))
    res = {"rel_L2": rel, "max_abs": float(np.abs(up - ue).max()), "wall_s": time.time() - t0,
           "loss_hist": hist, "library_rel_L2": lib["rel_L2"],
           "ratio_to_library": rel / lib["rel_L2"]}
    res["G0c"] = "PASS" if rel <= 2 * lib["rel_L2"] else "FAIL"
    json.dump(res, open(OUT, "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
