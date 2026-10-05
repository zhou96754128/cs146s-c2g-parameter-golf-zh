"""L3 显著性门禁：实现 + 自检 + 一次真实的可行性结论。

L3 要求：提升 ≥0.005 nats 且 p<0.01（≥3 seeds）。
本文件把这条判据写成可执行的函数，并用**已知答案**的输入自检——
因为如果门禁本身算错，「通过」就毫无意义。

实现两种检验：
  1) 精确符号翻转置换检验（stdlib，无分布假设）；
  2) 配对 t 检验（用临界值表，stdlib，无 scipy）。

自检会得出一个对本方案有实际影响的结论（见 SELF_TEST_FINDINGS）：
  n=3 时精确置换检验的最小 p = 2/8 = 0.25，**永远达不到 p<0.01**；
  要让精确检验达到 p<0.01，最少需要 n≥8 个 seed（2/256=0.0078）。
所以「≥3 seed」只够做描述性报告，够不上 L3 的显著性声明。
"""
import argparse
import itertools
import json
import math
import statistics
import sys

# 双侧 alpha=0.01 的 t 临界值表（df -> t_crit）
T_CRIT_001 = {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707,
              7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169, 15: 2.947, 20: 2.845,
              30: 2.750, 60: 2.660, 120: 2.617, 10 ** 9: 2.576}


def sign_flip_p(deltas):
    """精确符号翻转置换检验的 p 值（双侧）。n<=20 才枚举（2^20=1e6 可接受）。"""
    n = len(deltas)
    assert 1 <= n <= 20, n
    obs = abs(statistics.fmean(deltas))
    hits = 0
    total = 1 << n
    for mask in range(total):
        s = 0.0
        for i in range(n):
            s += deltas[i] if (mask >> i) & 1 else -deltas[i]
        if abs(s / n) >= obs - 1e-12:
            hits += 1
    return hits / total, total


def paired_t(deltas):
    """配对 t 统计量 + 双侧 alpha=0.01 的临界值判定。"""
    n = len(deltas)
    m = statistics.fmean(deltas)
    if n < 2:
        return {"n": n, "mean": m, "t": None, "df": 0, "t_crit": None,
                "significant": None, "note": "n<2，无法做配对检验"}
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(n)
    t = float("inf") if se == 0 else m / se
    df = n - 1
    keys = sorted(T_CRIT_001)
    t_crit = T_CRIT_001[min(keys, key=lambda k: abs(k - df))]
    return {"n": n, "mean": round(m, 6), "stderr": round(se, 6),
            "t": round(t, 4), "df": df, "t_crit": t_crit,
            "significant": abs(t) > t_crit,
            "note": f"|t|>{t_crit} 才在 alpha=0.01 下单侧/双侧显著（n={n} 已欠功效）"}


def gate(deltas, min_effect=0.005, alpha=0.01):
    """L3 判据：效应量 ≥ min_effect **且** 统计显著。"""
    g = paired_t(deltas)
    p_exact, total = sign_flip_p(deltas)
    effect_ok = g["mean"] >= min_effect
    return {
        "deltas": deltas,
        "min_effect": min_effect,
        "mean_effect": g["mean"],
        "effect_ok": effect_ok,
        "paired_t": g,
        "exact_sign_flip": {"p": round(p_exact, 6), "enumerations": total,
                            "min_possible_p": round(2 / total, 6)},
        "alpha": alpha,
        "verdict": "PASS" if (effect_ok and g["significant"]) else "FAIL",
    }


def self_test():
    out = {}

    # 1) 精确置换：全为 1、n=3 -> 观测 |mean|=1，只有全正/全负两种命中
    p, total = sign_flip_p([1.0, 1.0, 1.0])
    assert (p, total) == (0.25, 8), (p, total)
    out["sign_flip_n3_min"] = {"p": p, "total": total}

    # 2) 精确置换：n=8，7 正 1 负 -> 命中 2*(1+8)=18 / 256
    v = [1.0] * 8
    v[3] = -1.0
    p2, total2 = sign_flip_p(v)
    assert (p2, total2) == (18 / 256, 256), (p2, total2)
    out["sign_flip_n8_7pos1neg"] = {"p": round(p2, 6), "total": total2}

    # 3) 手算可验证的 t：均值 0.012，sd 0.002 -> t = 10.392 > 9.925(df=2) -> 显著
    g3 = gate([0.010, 0.012, 0.014])
    assert g3["effect_ok"] and g3["paired_t"]["significant"], g3
    assert abs(g3["paired_t"]["t"] - 10.3923) < 0.001, g3["paired_t"]["t"]
    out["handcheck_pass_case"] = {"t": g3["paired_t"]["t"], "verdict": g3["verdict"]}

    # 4) 噪声大的三 seed：均值 0.00867 >= 0.005 但 t=2.457 < 9.925 -> 不显著
    g4 = gate([0.010, 0.002, 0.014])
    assert g4["effect_ok"] and not g4["paired_t"]["significant"], g4
    assert g4["verdict"] == "FAIL", g4["verdict"]
    out["noisy_case"] = {"mean": g4["paired_t"]["mean"], "t": g4["paired_t"]["t"],
                         "verdict": g4["verdict"]}

    # 5) 关键结论：n=3 时精确置换检验的最小 p = 0.25，够不上 p<0.01
    min_n_exact = None
    for n in range(2, 15):
        if 2 / (1 << n) < 0.01:
            min_n_exact = n
            break
    assert min_n_exact == 8, min_n_exact
    out["SELF_TEST_FINDINGS"] = {
        "exact_permutation_min_p_at_n3": 0.25,
        "min_seeds_for_exact_p_lt_0.01": min_n_exact,
        "implication": "「≥3 seed」不足以支撑 L3 的 p<0.01 声明；精确检验需 ≥8 seed，"
                       "或改用参数检验并承认欠功效。",
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="")
    a = ap.parse_args()
    if a.deltas:
        ds = [float(x) for x in a.deltas.split(",")]
        print(json.dumps(gate(ds), ensure_ascii=False, indent=2))
        sys.exit(0)
    r = self_test()
    print("[significance] self-test PASS（实现与临界值表均已手工核验）")
    print(json.dumps(r, ensure_ascii=False, indent=2))
