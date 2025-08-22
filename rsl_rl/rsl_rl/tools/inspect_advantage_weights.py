#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import pickle
import numpy as np

def pct(x, q):
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, q))

def main():
    parser = argparse.ArgumentParser(description="Inspect advantages/weights saved in trajectories.")
    parser.add_argument("data_dir", help="Path to dataset directory containing trajectory_*/traj_*.pickle")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to scan (0 = all)")
    args = parser.parse_args()

    traj_dirs = [d for d in os.listdir(args.data_dir) if d.startswith("trajectory_")]
    traj_dirs.sort(key=lambda d: int(d.split("_")[1]))

    n_files = 0
    adv_all = []
    adv_pos_all = []
    values_all = []

    for td in traj_dirs:
        tdir = osp.join(args.data_dir, td)
        files = [f for f in os.listdir(tdir) if f.startswith("traj_") and f.endswith(".pickle")]
        files.sort()
        for f in files:
            with open(osp.join(tdir, f), "rb") as fh:
                traj = pickle.load(fh)
            # Optional keys
            values = traj.get("values", None)
            adv = traj.get("advantages", None)
            adv_pos = traj.get("positive_advantages", None)

            if values is not None:
                values_all.append(values.reshape(-1))
            if adv is not None:
                adv_all.append(adv.reshape(-1))
            if adv_pos is not None:
                adv_pos_all.append(adv_pos.reshape(-1))

            n_files += 1
            if args.limit and n_files >= args.limit:
                break
        if args.limit and n_files >= args.limit:
            break

    def cat_or_empty(lst):
        if not lst:
            return np.array([], dtype=np.float32)
        return np.concatenate(lst, axis=0)

    values_all = cat_or_empty(values_all)
    adv_all = cat_or_empty(adv_all)
    adv_pos_all = cat_or_empty(adv_pos_all)

    print("=== Advantage/Weight Inspection ===")
    print(f"Scanned files: {n_files}")
    print(f"values shape: {values_all.shape}, advantages shape: {adv_all.shape}, pos_adv shape: {adv_pos_all.shape}")

    if values_all.size:
        print("-- Teacher values --")
        print(f"mean={values_all.mean():.5f} std={values_all.std():.5f} min={values_all.min():.5f} max={values_all.max():.5f}")

    if adv_all.size:
        print("-- Advantages (GAE) --")
        pos_frac = float((adv_all > 0).mean())
        print(f"pos_frac={pos_frac:.3f} p50={pct(adv_all,50):.5f} p90={pct(adv_all,90):.5f} p95={pct(adv_all,95):.5f}")

    if adv_pos_all.size:
        print("-- Positive Advantages --")
        p95 = pct(adv_pos_all, 95)
        weights = adv_pos_all / (p95 + 1e-8)
        weights = np.clip(weights, 0.0, 1.0)
        print(f"p50={pct(adv_pos_all,50):.5f} p90={pct(adv_pos_all,90):.5f} p95={p95:.5f}")
        print("-- Derived Weights (A+ / P95, clamped [0,1]) --")
        print(f"mean={weights.mean():.5f} std={weights.std():.5f} min={weights.min():.5f} max={weights.max():.5f}")
        print(f"w>0.7 frac={float((weights>0.7).mean()):.3f} w>0.9 frac={float((weights>0.9).mean()):.3f}")

    if not adv_all.size and not adv_pos_all.size:
        print("No advantages found in trajectories. Ensure collect saved 'advantages'/'positive_advantages'.")

if __name__ == "__main__":
    main()
