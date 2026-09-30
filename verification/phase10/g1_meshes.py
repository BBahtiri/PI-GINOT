#!/usr/bin/env python3
"""
Phase 10, gate G1 -- the weak-form meshes of every geometry.

For every bank, validation, in-range and out-of-range geometry of the five
families, at the energy form's budget (~1400 triangles) and at a coarse level
(~350), checks:

  * det J > 0 at every quadrature point of every quad (Gauss-Legendre 6x6) and
    sum w det J = the triangulation's area (1e-12 relative);
  * the boundary edges close: every boundary node has exactly two boundary
    edges and they form closed loops; every edge has a segment name;
  * outward normals: sum over edges of n ds = 0 and of (x . n) ds = 2 * area
    (1e-12 relative), which fails for any inward or missing edge;
  * the quad boundary half-edges tile the triangle boundary edges
    (sum of half-edge lengths = boundary length);
  * P1 test DOFs dropped by C2 = the FEM's Dirichlet set, and every Dirichlet
    DOF lies on a segment flagged Dirichlet in that component;
  * reported, not gated: the largest turn between consecutive chord normals on
    each segment (the chord-vs-arc normal error is half of it).

    python -m verification.phase10.g1_meshes
"""
import json
import sys
from collections import defaultdict

import numpy as np

from geometry.vpinn_mesh import base_triangulation, boundary, dirichlet_dofs, split_to_quads
from physics.vpinn.mapping import bilinear
from physics.vpinn.quadrature import square_rule

OUT = "verification/results/phase10/g1.json"


def geometry_sets(fam_name):
    if fam_name == "dogbone":
        from geometry.banks import train_bank_params, val_bank_params
        from verification.heldout_sets import param_sets
        ir, oor, _ = param_sets()
        return {"bank": train_bank_params(64), "val": val_bank_params(2), "IR": ir, "OOR": oor}
    from geometry.families import FAMILIES
    f = FAMILIES[fam_name]
    return {"bank": f.bank(64), "val": f.val(2), "IR": f.in_range(), "OOR": f.out_of_range()}


def check(fam_name, p, n_elem):
    m = base_triangulation(fam_name, p, n_elem)
    b = boundary(fam_name, m)
    q = split_to_quads(m, b)
    xi, eta, w = square_rule(6)
    _, _, det = bilinear(q.cell_xy, xi, eta)
    area = float(m.areas.sum())
    r = {"n_tri": int(m.n_elem), "n_quad": int(len(q.cells)),
         "det_min_rel": float(det.min() / det.max()),
         "area_err": abs(float((w[None] * det).sum()) - area) / area}
    # closure
    deg = defaultdict(int)
    for a_, b_ in b.edges.tolist():
        deg[a_] += 1
        deg[b_] += 1
    r["bnd_deg_ok"] = all(v == 2 for v in deg.values())
    nxt = {a_: b_ for a_, b_ in b.edges.tolist()}
    seen, loops = set(), 0
    for s in nxt:
        if s in seen:
            continue
        loops += 1
        k = s
        while k not in seen:
            seen.add(k)
            k = nxt[k]
    r["loops"] = loops
    r["unnamed_edges"] = int((b.seg < 0).sum())
    ds = b.length[:, None] * b.normal
    mids = 0.5 * (m.nodes[b.edges[:, 0]] + m.nodes[b.edges[:, 1]])
    r["sum_n_ds"] = float(np.abs(ds.sum(0)).max() / b.length.sum())
    r["xn_err"] = abs(float((mids * ds).sum()) - 2 * area) / (2 * area)
    half = np.hypot(*(np.diff(q.cell_xy[q.bcell][:, [0, 1]], axis=1)[:, 0].T))  # ref edge 0 length
    e3 = np.hypot(*(q.cell_xy[q.bcell][:, 0] - q.cell_xy[q.bcell][:, 3]).T)
    hl = np.where(q.bref == 0, half, e3)
    r["halfedge_len_err"] = abs(float(hl.sum()) - float(b.length.sum())) / float(b.length.sum())
    # Dirichlet bookkeeping
    con = dirichlet_dofs(fam_name, m)
    seg_nodes = {i: set(np.asarray(m.boundary[s]).tolist()) for i, s in enumerate(b.names)}
    derived = set()
    for i in range(len(b.names)):
        for c in range(2):
            if b.dirichlet[i, c]:
                derived |= {2 * n + c for n in seg_nodes[i]}
    r["dirichlet_match"] = derived == set(con.tolist())
    # chord turn per segment
    turn = {}
    for i, s in enumerate(b.names):
        e = np.where(b.seg == i)[0]
        if len(e) < 2:
            continue
        nrm = b.normal[e]
        # order along the loop
        order = [e[0]]
        byfirst = {int(b.edges[k, 0]): k for k in e}
        while True:
            k = byfirst.get(int(b.edges[order[-1], 1]))
            if k is None or k in order:
                break
            order.append(k)
        nn = b.normal[order]
        ang = np.arccos(np.clip((nn[1:] * nn[:-1]).sum(1), -1, 1)) if len(order) > 1 else [0]
        turn[s] = float(np.max(ang))
    r["max_chord_turn_rad"] = turn
    r["pass"] = bool(r["det_min_rel"] > 0 and r["area_err"] < 1e-12 and r["bnd_deg_ok"]
                     and r["loops"] == 1 and r["unnamed_edges"] == 0 and r["sum_n_ds"] < 1e-12
                     and r["xn_err"] < 1e-12 and r["halfedge_len_err"] < 1e-12
                     and r["dirichlet_match"])
    return r


def main():
    fams = ["dogbone", "open_hole", "inclusion", "double_notch", "single_notch"]
    out, allpass = {}, True
    for fam in fams:
        sets = geometry_sets(fam)
        for level, n_elem in (("B", 1400), ("coarse", 350)):
            rows = []
            for split, plist in sets.items():
                for i, p in enumerate(plist):
                    r = check(fam, p, n_elem)
                    r["set"], r["i"] = split, i
                    rows.append(r)
            ok = all(r["pass"] for r in rows)
            allpass &= ok
            worst = {k: (max(r[k] for r in rows) if k != "det_min_rel" else min(r[k] for r in rows))
                     for k in ("det_min_rel", "area_err", "sum_n_ds", "xn_err", "halfedge_len_err")}
            turns = defaultdict(float)
            for r in rows:
                for s, v in r["max_chord_turn_rad"].items():
                    turns[s] = max(turns[s], v)
            out[f"{fam}|{level}"] = {"n_geometries": len(rows), "pass": ok,
                                    "failures": [(r["set"], r["i"]) for r in rows if not r["pass"]],
                                    "n_tri_range": [min(r["n_tri"] for r in rows), max(r["n_tri"] for r in rows)],
                                    "worst": worst, "max_chord_turn_rad": dict(turns),
                                    "loops": sorted({r["loops"] for r in rows}),
                                    "dirichlet_match": all(r["dirichlet_match"] for r in rows)}
            print(fam, level, "PASS" if ok else "FAIL", out[f"{fam}|{level}"]["n_tri_range"],
                  {k: f"{v:.1e}" for k, v in worst.items()},
                  {k: round(v, 3) for k, v in turns.items()},
                  "failures:", out[f"{fam}|{level}"]["failures"][:5], flush=True)
    out["G1"] = "PASS" if allpass else "FAIL"
    json.dump(out, open(OUT, "w"), indent=1)
    print("G1:", out["G1"])
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
