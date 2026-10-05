"""BPB (Bits-Per-Byte) 独立重实现 + 自测。

定义（与 C2G 附录 A 一致）：
    BPB = [ -Σ_i ln p(token_i) ] / (ln 2 * 总字节数)
说明：按「原始 UTF-8 字节」归一，因此与 vocab size 无关；这是它比 perplexity 公平的原因。

本文件不依赖任何挑战方代码，纯 stdlib。自测通过即说明我们的评测口径可复现。
"""
import math

LN2 = math.log(2)


def bpb_from_token_logprobs(token_logprobs, token_bytes):
    """token_logprobs: 每个 token 的自然对数概率；token_bytes: 对应 token 的字节长度。"""
    total_nats = -sum(token_logprobs)
    total_bytes = sum(token_bytes)
    if total_bytes <= 0:
        raise ValueError("total_bytes must be > 0")
    return (total_nats / LN2) / total_bytes


def bpb_from_probs(probs, token_bytes):
    return bpb_from_token_logprobs([math.log(p) for p in probs], token_bytes)


def self_test():
    results = {}

    # 1) 256 字节词表上的均匀分布 => 恰好 8.0 bits/byte
    lp = math.log(1.0 / 256)
    n = 1000
    v = bpb_from_token_logprobs([lp] * n, [1] * n)
    assert abs(v - 8.0) < 1e-9, v
    results["uniform_256bytes"] = round(v, 6)

    # 2) 尖峰分布（正确字节 90% 概率）=> 必须 < 8.0
    v2 = bpb_from_probs([0.9] * n, [1] * n)
    assert v2 < 8.0, v2
    results["peaked_90pct"] = round(v2, 6)

    # 3) 多字节 token 按「字节数」归一：4 个 token 各 3 字节、每个 p=0.5
    #    -log2(0.5)=1 bit/token => 4 bits / 12 bytes
    v3 = bpb_from_probs([0.5, 0.5, 0.5, 0.5], [3, 3, 3, 3])
    assert abs(v3 - 4.0 / 12.0) < 1e-12, v3
    results["multibyte_norm"] = round(v3, 6)

    # 4) tokenizer 无关性：同一段 4 字节文本的两种切分，只要「联合概率」与
    #    「总字节数」相同，BPB 必须一致。
    #    A: 1 个 token 覆盖 4 字节, p=0.25            -> 总 nats = -ln0.25
    #    B: 2 个 token 各 2 字节, p 各 0.5(联合 0.25)  -> 总 nats = -ln0.25
    a = bpb_from_probs([0.25], [4])
    b = bpb_from_probs([0.5, 0.5], [2, 2])
    assert abs(a - b) < 1e-12, (a, b)
    results["tokenizer_invariance"] = round(a, 6)

    return results


if __name__ == "__main__":
    r = self_test()
    print("[bpb] self-test PASS")
    for k, v in r.items():
        print("   -", k, "=", v)
