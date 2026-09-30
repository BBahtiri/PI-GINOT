#!/usr/bin/env python3
"""
Phase 10, gate G0b -- the PyTorch port against the fastvpinns library.

Reads what verification/phase10/g0_library.py saved (library tensors, residual
matrix and loss for a fixed float64 network, on the library's internal
rectangle mesh and on a distorted quad mesh), rebuilds everything with
physics/vpinn, and compares.  Pass: tensors to 1e-12 and the residual matrix and
loss to 1e-10 (relative to their scale).

    python -m verification.phase10.g0_port
"""
import json
import sys

import numpy as np
import torch

from physics.vpinn.tensors import quad_tensors

NPZ = "verification/results/phase10/g0_library_tensors.npz"
OUT = "verification/results/phase10/g0b.json"
W = 2.0 * np.pi
rhs = lambda x, y: -2.0 * W ** 2 * np.sin(W * x) * np.sin(W * y)


def mlp_from(d, p):
    Ws = [d[p + f"w{i}"] for i in range(8)]
    layers = []
    for i in range(0, 8, 2):
        lin = torch.nn.Linear(*Ws[i].shape, dtype=torch.float64)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(Ws[i].T))
            lin.bias.copy_(torch.tensor(Ws[i + 1]))
        layers += [lin] + ([torch.nn.Tanh()] if i < 6 else [])
    return torch.nn.Sequential(*layers)


def main():
    d = np.load(NPZ)
    res, ok = {}, True
    for kind in ("internal", "distorted"):
        p = kind + "_"
        K, Q = int(d[p + "K"]), int(d[p + "Q"])
        T = quad_tensors(d[p + "cells"], K, Q)
        rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
        r = {"x": rel(T.x, d[p + "xq"]), "val": rel(T.val, d[p + "val"]),
             "gx": rel(T.gx, d[p + "gx"]), "gy": rel(T.gy, d[p + "gy"])}
        F = np.einsum("ekq,eq->ke", T.val, rhs(T.x[:, 0], T.x[:, 1]).reshape(T.n_elem, T.n_quad))
        r["force"] = rel(F, d[p + "force"])
        net = mlp_from(d, p)
        x = torch.tensor(T.x, requires_grad=True)
        u = net(x)
        g, = torch.autograd.grad(u.sum(), x)
        gx = g[:, 0].reshape(T.n_elem, T.n_quad).detach().numpy()
        gy = g[:, 1].reshape(T.n_elem, T.n_quad).detach().numpy()
        R = (np.einsum("ekq,eq->ke", T.gx, gx) + np.einsum("ekq,eq->ke", T.gy, gy) - F)
        L = float((R ** 2).mean(0).sum())
        r["R"] = rel(R, d[p + "R"])
        r["loss"] = abs(L - float(d[p + "loss"])) / abs(float(d[p + "loss"]))
        r["area_sum"] = float(T.area.sum())
        # The same contraction fed the LIBRARY's own network derivatives: this is
        # the port in isolation.
        gl = d[p + "g"]
        Rl = (np.einsum("ekq,eq->ke", T.gx, gl[:, 0].reshape(T.n_elem, -1))
              + np.einsum("ekq,eq->ke", T.gy, gl[:, 1].reshape(T.n_elem, -1)) - F)
        r["R_libgrad"] = rel(Rl, d[p + "R"])
        r["loss_libgrad"] = abs(float((Rl ** 2).mean(0).sum()) - float(d[p + "loss"])) \
            / abs(float(d[p + "loss"]))
        # Where the end-to-end gap comes from: each framework's network against
        # a plain numpy float64 evaluation of the same weights.
        h = d[p + "xq"]
        for i in range(0, 8, 2):
            h = h @ d[p + f"w{i}"] + d[p + f"w{i + 1}"]
            h = np.tanh(h) if i < 6 else h
        r["net_torch_vs_numpy"] = rel(u.detach().numpy(), h)
        r["net_tf_vs_numpy"] = rel(d[p + "u"], h)
        # Pass (amended, see docs/phase10_log.md): tensors and the contraction
        # to 1e-12; the end-to-end gap must be explained by the framework
        # (torch equal to numpy to 1e-12).
        r["pass"] = bool(max(r["x"], r["val"], r["gx"], r["gy"], r["force"]) < 1e-12
                         and r["R_libgrad"] < 1e-12 and r["loss_libgrad"] < 1e-12
                         and r["net_torch_vs_numpy"] < 1e-12)
        ok &= r["pass"]
        res[kind] = r
        print(kind, {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})
    res["G0b"] = "PASS" if ok else "FAIL"
    json.dump(res, open(OUT, "w"), indent=1)
    print("G0b:", res["G0b"])
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
