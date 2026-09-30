#!/usr/bin/env python3
"""Phase 10, G4b -- summary table of every run (registered and diagnostic)."""
import glob
import json
import os

ROOT = "verification/results/phase10/g4b"


def main():
    rows = {}
    for d in sorted(glob.glob(f"{ROOT}/*/result.json")):
        name = d.split("/")[-2]
        if name.startswith("timing"):
            continue
        r = json.load(open(d))
        mon = r["monitor"]
        traj = []
        for m in mon:
            g = m["geoms"][0]
            traj.append({"step": m["step"], "gap": g["gap"], "err": g.get("err"), "vm": g.get("vm"),
                         "ratio": (g.get("res") or {}).get("ratio"), "Jmin": g["Jmin"]})
        rows[name] = {"s_per_step": r["s_per_step"], "clipped_frac": r["clipped_frac"],
                      "peak_rss_gb": r["peak_rss_gb"], "steps": r["steps"], "traj": traj}
    out = {"rows": rows}
    verdict = {}
    for fam in ("open_hole", "double_notch"):
        gB = rows.get(f"B_{fam}", {}).get("traj", [{}])[-1].get("gap")
        for arm in ("A", "B", "C1", "C1n", "C2", "C2I"):
            k = f"{arm}_{fam}"
            if k not in rows or gB is None:
                continue
            t = rows[k]["traj"]
            ok_i = t[-1]["gap"] < t[0]["gap"] and t[-1]["gap"] <= 3 * gB
            ok_ii = t[-1]["ratio"] is None or t[-1]["ratio"] <= 2.0
            verdict[k] = {"i": bool(ok_i), "ii": bool(ok_ii), "gap_over_B": t[-1]["gap"] / gB,
                          "G4b": "PASS" if ok_i and ok_ii else "FAIL"}
    out["verdict"] = verdict
    os.makedirs("docs/phase10_gates", exist_ok=True)
    json.dump(out, open("docs/phase10_gates/g4b.json", "w"), indent=1)
    for k, v in rows.items():
        t = v["traj"][-1]
        vd = verdict.get(k, {})
        print(f"{k:34s} {v['s_per_step']:5.2f} s/step  gap {v['traj'][0]['gap']:+.4f} -> {t['gap']:+.5f}"
              f"  Kt err {t['err']:+.3f}  vm {t['vm']:.3f}  ratio {t['ratio'] if t['ratio'] is None else round(t['ratio'], 2)}"
              f"  clip {v['clipped_frac']:.2f}  {vd.get('G4b', '')}"
              f"{' (' + format(vd['gap_over_B'], '.1f') + 'x B)' if vd else ''}")


if __name__ == "__main__":
    main()
