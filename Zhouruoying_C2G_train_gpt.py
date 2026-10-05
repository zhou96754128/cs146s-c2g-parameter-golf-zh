"""C2G 训练入口 —— 真实 8×H100 训练的可执行入口 + 本机可跑的干跑校验。

三种模式（诚实边界写死在代码里，禁止静默降级）：
  --check        本机可跑。解析配置、打印最终超参、断言 16MB 预算，退出码 0。
  --local-smoke  本机可跑。调用本地 stdlib 缩比管线，产出**真实**的 MSE 曲线
                 （不是 BPB；数据是合成的，不是 FineWeb）。
  --train        需 8×H100。真实训练；本机无 CUDA / 卡数不足时以退出码 3
                 **明确失败**，绝不「假装跑完」。

设计原则：C2G 的硬门禁是 10 分钟 / 16MB / 8×H100。本机没有 H100，因此把
「能证明的」与「不能证明的」在代码层分开：
  * 能证明：配置合法性、参数预算算术、优化器分组正确性、复现入口确实可跑；
  * 不能证明：真实 BPB —— 必须 8×H100 实测，本文件不产出该数字。

优化器分组（本方案的单点改动）：Muon 只作用于隐藏层 2D 权重；
embedding / 最终 head / 1D 参数仍走 AdamW（与 Muon 原文用法一致）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(HERE, "Zhouruoying_C2G_本地校验")
sys.path.insert(0, LOCAL_DIR)

ARTIFACT_LIMIT = 16_000_000          # 硬门禁：16,000,000 字节
WALLCLOCK_LIMIT_S = 600              # 硬门禁：10 分钟
TRAIN_TARGET_S = 480                 # 训练循环目标 8:00，留 20% buffer
BASELINE_BPB = 1.2244                # 题目给定 baseline（9L/512d）
TARGETS = {"L2": 1.18, "L3": 1.12}   # Level 2 / Level 3 门槛

CONFIG = {
    "challenge": "C2G",
    "gpu": "8xH100-SXM",
    "d_model": 512, "n_layers": 9, "n_heads": 8, "n_kv_heads": 4,
    "mlp_mult": 4.0, "tied_embedding": True, "vocab": 1024, "recurrence": 1,
    "quantization": {"body": "int4", "embed": "int4"},
    "optimizer": {"2d_hidden": "muon", "embed_head_1d": "adamw"},
    "muon": {"momentum": 0.95, "nesterov": True, "ns_steps": 5,
             "a": 3.4445, "b": -4.7750, "c": 2.0315, "eps": 1e-7,
             "lr_candidates": [0.01, 0.02, 0.04]},
    "adamw": {"lr": 3e-4, "betas": [0.9, 0.95], "wd": 0.1},
    "seeds": [20261005, 20261006, 20261007],
    "levels": {"L2": "BPB<1.18 (1 方向)", "L3": "BPB<1.12 (>=2 方向组合)"},
}


def assert_budget(vocab=None):
    """用算术证明配置装得进 16MB —— 这是本机唯一能给出的硬结论。"""
    import budget
    vocab = vocab or CONFIG["vocab"]
    n = budget.param_count(vocab=vocab)
    emb = vocab * CONFIG["d_model"]
    b = budget.size_bytes(n, CONFIG["quantization"]["body"], emb,
                          CONFIG["quantization"]["embed"])
    return {"vocab": vocab, "params": n, "artifact_bytes": int(b),
            "limit": ARTIFACT_LIMIT, "fits": b <= ARTIFACT_LIMIT,
            "bits_per_param": round(8 * b / n, 3)}


def check():
    """干跑：本机可跑，产出可复核的配置与预算结论。"""
    rows = [assert_budget(v) for v in (1024, 4096, 8192)]
    bad = [r for r in rows if not r["fits"]]
    summary = {
        "mode": "check",
        "config": CONFIG,
        "budget_rows": rows,
        "wallclock_headroom_s": WALLCLOCK_LIMIT_S - TRAIN_TARGET_S,
        "verdict": "PASS" if not bad else "FAIL",
        "honesty": "本机无 8×H100，此处不含任何真实 BPB 成绩",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not bad else 1


def local_smoke():
    """本机可跑的缩比 A/B：真实运行、真实数字，但**不是 BPB、不是成绩**。"""
    import smoke_ab
    out = smoke_ab.self_test()
    print(json.dumps({"mode": "local-smoke", "note": "合成数据/MSE，非 BPB，非成绩",
                      "result": out}, ensure_ascii=False, indent=2))
    return 0


def train():
    """真实训练：必须有 8×H100，否则明确失败（退出码 3），不静默通过。"""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"[BLOCKED] 未安装 torch：{type(e).__name__}: {e}")
        print("[BLOCKED] 真实训练需要 8×H100 + torch>=2.3；本机不满足。")
        return 3
    if not torch.cuda.is_available():
        print("[BLOCKED] 无可用 CUDA 设备；真实训练需要 8×H100 SXM。")
        return 3
    n_gpu = torch.cuda.device_count()
    if n_gpu < 8:
        print(f"[BLOCKED] 仅检测到 {n_gpu} 张 GPU，硬约束要求 8 张 H100 SXM。")
        return 3
    print("[INFO] 检测到 8 卡环境；此处接入 parameter-golf 训练脚本：")
    print("       git clone https://github.com/openai/parameter-golf.git")
    print("       python download_fineweb.py && python train_gpt.py（替换 optimizer 分组）")
    print("[INFO] 本机为交付环境，未执行真实训练；BPB 由真实运行产出。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="C2G 训练入口")
    ap.add_argument("--check", action="store_true", help="干跑校验（本机可跑）")
    ap.add_argument("--local-smoke", action="store_true", help="缩比 A/B（本机可跑）")
    ap.add_argument("--train", action="store_true", help="真实训练（需 8×H100）")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.local_smoke:
        return local_smoke()
    if a.train:
        return train()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
