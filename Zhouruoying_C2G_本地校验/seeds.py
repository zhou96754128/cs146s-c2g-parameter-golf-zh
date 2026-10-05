"""3-seed 复跑器 —— 让「≥3 seed」这条硬要求在**本机**也有真实、可复核的落盘证据。

诚实边界（与 smoke_ab 一致）：
  跑的是合成数据 + MSE 的缩比管线，**不是 FineWeb，不是 BPB**，
  因此**不构成 L2/L3 成绩证据**。它证明的是两件真事：
    ① 固定 seed 完全可复现（同 seed 两次运行 loss 逐位相同）；
    ② 我们的对照统计链路（逐 seed 记录 → 均值 ± stderr → 配对差）确实能跑出数字。

用法：python3 seeds.py [--outdir ../Zhouruoying_C2G_logs] [--quiet]
"""
import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_ab  # noqa: E402

SEEDS = [20261005, 20261006, 20261007]
DEFAULT_OUTDIR = os.path.join(os.path.dirname(HERE), "Zhouruoying_C2G_logs")


def run_seed(seed):
    """换 seed 同时改变数据生成与权重初始化（smoke_ab.SEED 是唯一来源）。"""
    smoke_ab.SEED = seed
    a = smoke_ab.train("adamw")
    b = smoke_ab.train("muon")
    return {
        "seed": seed,
        "adamw": a,
        "muon": b,
        "delta_reduced_pct": round(b["reduced_by_pct"] - a["reduced_by_pct"], 4),
        "delta_loss_last": round(b["loss_last"] - a["loss_last"], 8),
    }


def mean_stderr(xs):
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, statistics.stdev(xs) / math.sqrt(len(xs))


def repeatability(seed):
    """同 seed 连跑两次必须完全一致 —— 这是「可复现」的硬断言，不是感觉。"""
    r1 = run_seed(seed)
    r2 = run_seed(seed)
    return {
        "seed": seed,
        "adamw_loss_last_equal": r1["adamw"]["loss_last"] == r2["adamw"]["loss_last"],
        "muon_loss_last_equal": r1["muon"]["loss_last"] == r2["muon"]["loss_last"],
        "adamw_loss_last": r1["adamw"]["loss_last"],
        "muon_loss_last": r1["muon"]["loss_last"],
    }


def main(quiet=False, outdir=None):
    outdir = outdir or DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)

    rows = [run_seed(s) for s in SEEDS]
    for r in rows:
        with open(os.path.join(outdir, f"seed_{r['seed']}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        # 每个 seed 留一份人类可读日志，便于人工复核
        with open(os.path.join(outdir, f"seed_{r['seed']}.log"), "w",
                  encoding="utf-8") as f:
            f.write(f"[seed {r['seed']}] 缩比 A/B（合成数据/MSE，非 BPB）\n")
            for k in ("adamw", "muon"):
                v = r[k]
                f.write(f"  {k:6s} loss_first={v['loss_first']:.6f} "
                        f"loss_last={v['loss_last']:.6f} "
                        f"reduced={v['reduced_by_pct']:.2f}% "
                        f"wallclock={v['wallclock_s']:.3f}s\n")
            f.write(f"  delta_reduced_pct={r['delta_reduced_pct']:+.4f}\n")

    a_m, a_se = mean_stderr([r["adamw"]["reduced_by_pct"] for r in rows])
    b_m, b_se = mean_stderr([r["muon"]["reduced_by_pct"] for r in rows])
    d_m, d_se = mean_stderr([r["delta_reduced_pct"] for r in rows])

    rep = repeatability(SEEDS[0])

    summary = {
        "kind": "local-scaled-3seed",
        "is_evidence_for_L2_L3": False,
        "why_not": "合成数据 + MSE 缩比，不是 FineWeb/BPB；仅证明管线可跑且可复现",
        "seeds": SEEDS,
        "per_seed": [{k: r[k] for k in ("seed", "delta_reduced_pct", "delta_loss_last")}
                     for r in rows],
        "adamw_reduced_pct": {"mean": round(a_m, 4), "stderr": round(a_se, 4)},
        "muon_reduced_pct": {"mean": round(b_m, 4), "stderr": round(b_se, 4)},
        "delta_reduced_pct": {"mean": round(d_m, 4), "stderr": round(d_se, 4)},
        "repeatability_same_seed": rep,
        "outdir": outdir,
    }
    with open(os.path.join(outdir, "seeds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ok = (rep["adamw_loss_last_equal"] and rep["muon_loss_last_equal"]
          and a_m > 0 and b_m > 0 and len(rows) >= 3)
    summary["verdict"] = "PASS" if ok else "FAIL"
    with open(os.path.join(outdir, "seeds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def self_test():
    """run_all 用：跑 3 seed 并断言可复现 + 两条路径都在学。"""
    rc = main(quiet=True)
    assert rc == 0, "seeds self-test failed"
    with open(os.path.join(DEFAULT_OUTDIR, "seeds_summary.json"), encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C2G 3-seed 缩比复跑")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    sys.exit(main(quiet=args.quiet, outdir=args.outdir))
