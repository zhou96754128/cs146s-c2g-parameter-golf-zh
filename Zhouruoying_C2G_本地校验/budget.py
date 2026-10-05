"""16MB 参数预算计算器 —— 用来验证「配置是否装得下」。

为什么要它：C2G 的 16,000,000 字节是硬门禁。挑战文本给出的 baseline 只说了
「9 层 / 512 维 / tied embedding / 4 KV heads」，并未给出 MLP 比例与量化方案。
因此本文件是一个**透明的口径计算器**（所有假设都写在公式里），用于：

  1) 用算术证明 16MB 有多紧：该规模模型约 26.5M 参数（d=512, L=9, MLP×4, tied），
     bf16 要约 52MB、int8 要约 26MB —— 两者都超出 16MB；
     只有降到 ≤ ~4.8 bit/参数（int4 一档）才装得下。
  2) 校验我们自己的配置选择是否越界。

诚实边界：下列数值是**按公开公式的算术估算**，不是平台测量值；
「int4 量化是否掉点」本文件不回答，那必须靠真实训练实验。
"""
import json

BYTES_PER_PARAM = {"bf16": 2.0, "int8": 1.0, "int6": 0.75, "int4": 0.5}


def param_count(d_model=512, n_layers=9, vocab=1024, n_heads=8, n_kv_heads=4,
                mlp_mult=4.0, tied=True, recurrence=1):
    """标准 decoder-only Transformer 参数量（无 bias）。

    组成：
      每层 = attn(q,o: 2*d*d) + attn(k,v: 2*d*(d*n_kv/n_heads)) + mlp(2*d*(d*mlp_mult))
      x n_layers；再叠加 tying 后的 embedding（输入/输出共享一份）。
    recurrence（3-Layer Recurrence）复用权重，不增加参数量。
    """
    d = d_model
    kv_dim = d * n_kv_heads // n_heads
    per_layer = (2 * d * d) + (2 * d * kv_dim) + (2 * d * int(d * mlp_mult))
    body = per_layer * n_layers
    embed = vocab * d
    total = body + embed  # tied: 输入/输出共享
    return total


def size_bytes(n_params, dtype="bf16", embed_params=None, embed_dtype=None):
    """混合精度：embedding 可单独量化（body 与 embedding 可不同精度）。"""
    if embed_params is None or embed_dtype is None:
        return n_params * BYTES_PER_PARAM[dtype]
    other = n_params - embed_params
    return other * BYTES_PER_PARAM[dtype] + embed_params * BYTES_PER_PARAM[embed_dtype]


def check(limit=16_000_000):
    rows = []
    d = 512
    for vocab in (1024, 4096, 8192):
        n = param_count(vocab=vocab)
        emb = vocab * d
        for dtype, edtype in (("bf16", None), ("int8", None), ("int6", None),
                              ("int4", None), ("bf16", "int6")):
            b = size_bytes(n, dtype, emb, edtype) if edtype else size_bytes(n, dtype)
            rows.append({
                "vocab": vocab,
                "params_M": round(n / 1e6, 2),
                "body_dtype": dtype,
                "embed_dtype": edtype or dtype,
                "MB": round(b / 1e6, 2),
                "fits_16MB": b <= limit,
            })
    return rows


def self_test():
    rows = check()
    assert rows, "no rows"

    def pick(vocab, body, embed):
        return [r for r in rows
                if r["vocab"] == vocab and r["body_dtype"] == body
                and r["embed_dtype"] == embed]

    # 1) bf16 全精度必然装不下（26.5M 参数 -> ~52MB）
    bf16 = pick(1024, "bf16", "bf16")
    assert bf16 and not bf16[0]["fits_16MB"], bf16
    # 2) int8 也装不下（~26MB），说明「只量化一半」不够
    int8 = pick(1024, "int8", "int8")
    assert int8 and not int8[0]["fits_16MB"], int8
    # 3) int6 也不够（~19.9MB）
    int6 = pick(1024, "int6", "int6")
    assert int6 and not int6[0]["fits_16MB"], int6
    # 4) 只量化 embedding、body 仍 bf16 -> 依然超预算（澄清常见误解）
    mixed = pick(1024, "bf16", "int6")
    assert mixed and not mixed[0]["fits_16MB"], mixed
    # 5) int4 全量化必须装得下 -> 结论：16MB 强制 int4 级量化
    int4 = pick(1024, "int4", "int4")
    assert int4 and int4[0]["fits_16MB"], int4
    # 6) 更大词表（8192）在 int4 下仍应装得下
    big = pick(8192, "int4", "int4")
    assert big and big[0]["fits_16MB"], big

    return rows


if __name__ == "__main__":
    rows = self_test()
    print("[budget] self-test PASS —— 口径：9L/512d/tied/4KV/MLP x4")
    print(f"{'vocab':>6} | {'params(M)':>9} | {'body':>6} | {'embed':>6} | {'MB':>7} | fits16MB")
    for r in rows:
        print(f"{r['vocab']:>6} | {r['params_M']:>9} | {r['body_dtype']:>6} | "
              f"{r['embed_dtype']:>6} | {r['MB']:>7} | {r['fits_16MB']}")
