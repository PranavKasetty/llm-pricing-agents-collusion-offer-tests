"""Statistical analysis for Experiment 1 (Summary vs Raw History) and Experiment 2 (Expected-Damages Regulator).
"""

import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]

def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial test for McNemar table [[a, b], [c, d]]."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(b, n, 0.5).pvalue)

def analyze(path: Path):
    sessions = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            sessions[r["seed"]][r["version"]] = r

    seeds = sorted(sessions.keys())
    versions = ["R", "O", "R_reg", "O_reg", "R_summary", "O_summary"]

    print(f"Total sessions: {len(seeds)}")
    print("=" * 80)
    print(f"{'Version':<12} | {'Median Price (Mean ± SD)':<25} | {'Acc @ 1.776':<18} | {'Acc @ 1.70':<18}")
    print("-" * 80)

    summary_stats = {}
    for v in versions:
        prices = [sessions[s][v]["median_price"] for s in seeds]
        acc_1776 = sum(1 for s in seeds if sessions[s][v]["accepted"]["1.776"])
        acc_1700 = sum(1 for s in seeds if sessions[s][v]["accepted"]["1.7"])
        n = len(seeds)
        ci_1776 = wilson(acc_1776, n)
        ci_1700 = wilson(acc_1700, n)
        mean_p = np.mean(prices)
        std_p = np.std(prices, ddof=1)
        summary_stats[v] = {
            "prices": prices,
            "mean": mean_p,
            "std": std_p,
            "acc_1776": acc_1776,
            "acc_1700": acc_1700,
            "ci_1776": ci_1776,
            "ci_1700": ci_1700
        }
        print(f"{v:<12} | {mean_p:.4f} ± {std_p:.4f}          | {acc_1776}/{n} ({acc_1776/n*100:4.1f}% [{ci_1776[0]:.2f},{ci_1776[1]:.2f}]) | {acc_1700}/{n} ({acc_1700/n*100:4.1f}% [{ci_1700[0]:.2f},{ci_1700[1]:.2f}])")

    print("\n" + "=" * 80)
    print("PAIRED CONTRASTS (N = 30 pairs)")
    print("=" * 80)

    contrasts = [
        ("O vs R (Offer effect under raw history)", "O", "R"),
        ("O_reg vs R_reg (Offer effect under regulator)", "O_reg", "R_reg"),
        ("O_summary vs R_summary (Offer effect under summary)", "O_summary", "R_summary"),
        ("O_reg vs O (Regulator intervention effect on offer)", "O_reg", "O"),
        ("R_reg vs R (Regulator baseline effect on uninvited)", "R_reg", "R"),
        ("O_summary vs O (History interface effect on offer)", "O_summary", "O"),
        ("R_summary vs R (History interface effect on baseline)", "R_summary", "R"),
    ]

    for label, v1, v2 in contrasts:
        p1 = np.array([sessions[s][v1]["median_price"] for s in seeds])
        p2 = np.array([sessions[s][v2]["median_price"] for s in seeds])
        diff = p1 - p2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        # Wilcoxon
        w_stat, w_p = stats.wilcoxon(p1, p2, zero_method="wilcox") if not np.all(p1 == p2) else (0, 1.0)
        t_stat, t_p = stats.ttest_rel(p1, p2)

        # McNemar for acceptance @ 1.776
        # b: v1=True, v2=False; c: v1=False, v2=True
        b = sum(1 for s in seeds if sessions[s][v1]["accepted"]["1.776"] and not sessions[s][v2]["accepted"]["1.776"])
        c = sum(1 for s in seeds if not sessions[s][v1]["accepted"]["1.776"] and sessions[s][v2]["accepted"]["1.776"])
        mc_p = mcnemar(b, c)

        print(f"\n>>> {label}")
        print(f"    Mean Diff: {mean_diff:+.4f} ± {std_diff:.4f}")
        print(f"    Paired t-test: t = {t_stat:.3f}, p = {t_p:.4f}")
        print(f"    Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_p:.4f}")
        print(f"    Acceptance @ 1.776: {v1}={summary_stats[v1]['acc_1776']} vs {v2}={summary_stats[v2]['acc_1776']} (Discordant pairs: +{b}/-{c}, McNemar p = {mc_p:.4f})")

if __name__ == "__main__":
    analyze(Path("repo/data/fork_runs/p1_experiments/results.jsonl"))
