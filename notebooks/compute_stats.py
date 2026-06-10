#!/usr/bin/env python3
"""
compute_stats.py  --  correction-angle agreement statistics (CPU-only).

Reads a CSV of paired correction angles (one row per limb hemisphere) and
reports the numbers needed to compare against Przystalski et al. 2023:

    - n, mean / median / std / max ABSOLUTE error (degrees)
    - RMSE
    - % of cases within the clinical tolerance (default +/- 1.63 deg, Jiang et al.)
    - ICC(2,1)  (two-way random, absolute agreement, single rater) with 95% CI
    - Bland-Altman bias and 95% limits of agreement
    - Pearson r

The CSV must have two columns of angles in degrees. By default they are named
'gt_angle' (mean-observer ground truth) and 'pred_angle' (automatic method),
which is what compute_angle_pairs.py writes, but you can override the names.

Nothing here touches a GPU or the model; it runs on a laptop in milliseconds.

Usage
-----
    python compute_stats.py angle_pairs_cv.csv
    python compute_stats.py angle_pairs_cv.csv angle_pairs_test.csv
    python compute_stats.py mydata.csv --gt col_a --pred col_b --tolerance 1.63

ICC: 'pingouin' (the package the predecessor used) is preferred because it
returns an exact 95% CI. If pingouin is not installed the script falls back to
a manual ICC(2,1) point estimate and tells you to `pip install pingouin` for
the interval.
"""

import argparse
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- ICC
def icc21_manual(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    Parameters
    ----------
    ratings : array, shape (n_targets, k_raters)

    Returns
    -------
    Point estimate only (no CI). Follows Shrout & Fleiss (1979) /
    McGraw & Wong (1996), case (2,1).
    """
    n, k = ratings.shape
    grand = ratings.mean()
    row_means = ratings.mean(axis=1, keepdims=True)   # per target
    col_means = ratings.mean(axis=0, keepdims=True)   # per rater

    ss_total = ((ratings - grand) ** 2).sum()
    ss_row = k * ((row_means - grand) ** 2).sum()      # between targets
    ss_col = n * ((col_means - grand) ** 2).sum()      # between raters
    ss_err = ss_total - ss_row - ss_col

    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_row + (k - 1) * ms_err + (k / n) * (ms_col - ms_err)
    return float((ms_row - ms_err) / denom)


def compute_icc(gt: np.ndarray, pred: np.ndarray):
    """Return (icc_value, ci_low, ci_high, method_string).

    Tries pingouin (exact CI). Falls back to a manual point estimate.
    The two 'raters' are the mean-observer GT and the automatic method, so
    this reproduces the predecessor's 'MR vs FAM' comparison.
    """
    n = len(gt)
    try:
        import pingouin as pg
        long = pd.DataFrame({
            "target": list(range(n)) * 2,
            "rater": ["mean_observer"] * n + ["auto"] * n,
            "angle": np.concatenate([gt, pred]),
        })
        res = pg.intraclass_corr(
            data=long, targets="target", raters="rater", ratings="angle"
        )
        # Two-way random, absolute agreement, single measurement = ICC(2,1).
        # pingouin labels this 'ICC2' (<=0.5.x) or 'ICC(A,1)' (>=0.6.x).
        mask = res["Type"].isin(["ICC2", "ICC(A,1)"])
        row = res.loc[mask].iloc[0]
        ci_col = "CI95%" if "CI95%" in res.columns else "CI95"
        ci = row[ci_col]
        return float(row["ICC"]), float(ci[0]), float(ci[1]), "pingouin ICC(2,1)"
    except ImportError:
        val = icc21_manual(np.column_stack([gt, pred]))
        return val, np.nan, np.nan, "manual ICC(2,1) -- pip install pingouin for the 95% CI"


# ----------------------------------------------------------------------------- report
def report(df: pd.DataFrame, gt_col: str, pred_col: str, tolerance: float, label: str):
    gt = df[gt_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    diff = pred - gt                       # signed error (for Bland-Altman)
    abs_err = np.abs(diff)                 # absolute error (matches predecessor)

    icc, lo, hi, method = compute_icc(gt, pred)

    bias = diff.mean()
    sd_diff = diff.std(ddof=1)
    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff
    pearson = np.corrcoef(gt, pred)[0, 1]
    within = 100.0 * np.mean(abs_err <= tolerance)

    icc_str = f"{icc:.3f}" if not np.isnan(lo) else f"{icc:.3f} (no CI)"
    ci_str = "" if np.isnan(lo) else f" ({lo:.3f}-{hi:.3f} 95% CI)"

    print("=" * 64)
    print(f"  {label}   (n = {len(df)} limb hemispheres)")
    print("=" * 64)
    print("Absolute correction-angle error (degrees)")
    print(f"  mean    : {abs_err.mean():.4f}")
    print(f"  median  : {np.median(abs_err):.4f}")
    print(f"  std     : {abs_err.std(ddof=1):.4f}")
    print(f"  min     : {abs_err.min():.4f}")
    print(f"  max     : {abs_err.max():.4f}")
    print(f"  RMSE    : {np.sqrt(np.mean(diff ** 2)):.4f}")
    print(f"  within +/-{tolerance:.2f} deg : {within:.1f}%")
    print()
    print("Agreement")
    print(f"  ICC(2,1): {icc_str}{ci_str}   [{method}]")
    print(f"  Pearson r: {pearson:.4f}")
    print()
    print("Bland-Altman (predicted - GT)")
    print(f"  bias (mean diff)      : {bias:+.4f}")
    print(f"  95% limits of agreement: [{loa_low:+.4f}, {loa_high:+.4f}]")
    print()
    print("For reference, Przystalski et al. 2023 (MR vs FAM, same protocol):")
    print("  mean 0.5  | median 0.3  | max 2.76  | ICC 0.99 (0.98-0.99)")
    print("=" * 64)
    print()

    return {
        "label": label, "n": len(df),
        "mean": abs_err.mean(), "median": float(np.median(abs_err)),
        "std": abs_err.std(ddof=1), "max": abs_err.max(),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "within_tol_pct": within,
        "icc": icc, "icc_lo": lo, "icc_hi": hi,
        "ba_bias": bias, "ba_loa_low": loa_low, "ba_loa_high": loa_high,
        "pearson": pearson,
    }


def main():
    p = argparse.ArgumentParser(description="Correction-angle agreement statistics.")
    p.add_argument("csv", nargs="+", help="one or more angle-pairs CSV files")
    p.add_argument("--gt", default="gt_angle", help="ground-truth column name")
    p.add_argument("--pred", default="pred_angle", help="prediction column name")
    p.add_argument("--tolerance", type=float, default=1.63,
                   help="clinical tolerance in degrees (default 1.63, Jiang et al.)")
    p.add_argument("--out", default=None, help="optional CSV path to save the summary table")
    args = p.parse_args()

    summaries = []
    for path in args.csv:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"!! file not found: {path}", file=sys.stderr)
            continue
        for col in (args.gt, args.pred):
            if col not in df.columns:
                print(f"!! column '{col}' not in {path}; columns are {list(df.columns)}",
                      file=sys.stderr)
                break
        else:
            summaries.append(report(df, args.gt, args.pred, args.tolerance, label=path))

    if args.out and summaries:
        pd.DataFrame(summaries).to_csv(args.out, index=False)
        print(f"Summary table written to {args.out}")


if __name__ == "__main__":
    main()
