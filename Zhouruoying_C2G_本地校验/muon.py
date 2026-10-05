"""Muon 的 Newton–Schulz 正交化 —— 独立重实现 + 单测。

系数与步数取自 Muon 原文（Keller Jordan et al., 2024-12-08）：
    5 步迭代, a = 3.4445, b = -4.7750, c = 2.0315, eps = 1e-7
本文件**零第三方依赖**（只 stdlib），矩阵乘/转置/范数均为自实现：
目的是让任何一台装有 Python 3 的机器都能原样复现，无需安装 numpy。

判据说明（与 numpy 版等价但更直接）：
    半正交矩阵 X（m<=n）满足 X @ X.T ≈ I_m。
    因此用 Gram 缺陷 ‖X Xᵀ − I‖_max 度量「正交化是否把谱压平」，
    替代 SVD 奇异值比（numpy 的 SVD 属实现细节，非必需）。

诚实边界：这是**机制**验证，不是 BPB 成绩，也不是「Muon 一定更好」的证明。
"""
import math
import random

A_COEF, B_COEF, C_COEF = 3.4445, -4.7750, 2.0315
EPS = 1e-7
STEPS = 5
SEED = 20261005


# ---------- 纯 stdlib 线性代数小工具 ----------
def matmul(A, B):
    n, m = len(A), len(A[0])
    m2, p = len(B), len(B[0])
    assert m == m2, (m, m2)
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai, Oi = A[i], out[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                Oi[j] += a * Bk[j]
    return out


def transpose(A):
    return [list(col) for col in zip(*A)]


def fro(A):
    return math.sqrt(sum(v * v for row in A for v in row))


def gauss_matrix(rng, rows, cols, scale=1.0):
    return [[rng.gauss(0.0, 1.0) * scale for _ in range(cols)] for _ in range(rows)]


def gram_defect(X):
    """X(m<=n) 的半正交缺陷：max|X Xᵀ − I|。越小越接近正交。"""
    G = matmul(X, transpose(X))
    m = len(G)
    worst = 0.0
    diag_lo, diag_hi = float("inf"), float("-inf")
    for i in range(m):
        for j in range(m):
            tgt = 1.0 if i == j else 0.0
            d = abs(G[i][j] - tgt)
            if d > worst:
                worst = d
        diag_lo = min(diag_lo, G[i][i])
        diag_hi = max(diag_hi, G[i][i])
    return worst, diag_lo, diag_hi


# ---------- 待验证实现 ----------
def zeropower_via_newtonschulz5(G, steps=STEPS, eps=EPS):
    """把矩阵 G 的更新量正交化（近似 zero-power / 半正交）。"""
    a, b, c = A_COEF, B_COEF, C_COEF
    X = [row[:] for row in G]
    transposed = False
    if len(X) > len(X[0]):
        X = transpose(X)
        transposed = True
    nrm = fro(X) + eps
    X = [[v / nrm for v in row] for row in X]
    for _ in range(steps):
        AA = matmul(X, transpose(X))
        AA2 = matmul(AA, AA)
        B_ = [[b * AA[i][j] + c * AA2[i][j] for j in range(len(AA[0]))]
              for i in range(len(AA))]
        BX = matmul(B_, X)                      # 必须用「未缩放」的 X 计算 B@X
        X = [[a * X[i][j] + BX[i][j] for j in range(len(X[0]))]
             for i in range(len(X))]
    if transposed:
        X = transpose(X)
    return X


def self_test():
    rng = random.Random(SEED)
    results = {}

    # 1) 随机 64x32 矩阵：正交化前 Gram 缺陷很大，正交化后应显著变小
    G = gauss_matrix(rng, 64, 32)
    before, _, _ = gram_defect(transpose(G))  # 64x32 -> 取 m=min 侧对齐
    O = zeropower_via_newtonschulz5(G)
    after, diag_lo, diag_hi = gram_defect(transpose(O))
    results["gram_defect_before"] = round(before, 4)
    results["gram_defect_after"] = round(after, 4)
    assert after < before, (before, after)
    assert after < 0.6, f"orthogonalization weak: {after}"

    # 2) 半正交：Gram 对角线应集中在 1 附近
    results["gram_diag_min"] = round(diag_lo, 4)
    results["gram_diag_max"] = round(diag_hi, 4)
    assert 0.5 < diag_lo and diag_hi < 1.5, (diag_lo, diag_hi)

    # 3) 形状保持（含转置分支）：40x80 与 80x40
    for shape in [(40, 80), (80, 40)]:
        M = gauss_matrix(rng, shape[0], shape[1])
        out = zeropower_via_newtonschulz5(M)
        assert len(out) == shape[0] and len(out[0]) == shape[1], (shape,)
    results["shape_preserved"] = True

    # 4) 确定性：同输入同输出（逐位相等）
    O2 = zeropower_via_newtonschulz5(G)
    assert O == O2, "non-deterministic"
    results["deterministic"] = True

    # 5) 零矩阵不炸（eps 保护）
    Z = zeropower_via_newtonschulz5([[0.0] * 8 for _ in range(8)])
    assert all(math.isfinite(v) for row in Z for v in row), "NaN/Inf on zero input"
    results["zero_input_safe"] = True

    return results


if __name__ == "__main__":
    r = self_test()
    print("[muon] self-test PASS")
    for k, v in r.items():
        print("   -", k, "=", v)
