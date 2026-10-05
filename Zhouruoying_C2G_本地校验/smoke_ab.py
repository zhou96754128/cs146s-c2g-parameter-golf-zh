"""A/B 对照流水线的「机制」smoke test —— CPU + stdlib，无 GPU、无外网、无第三方依赖。

目的：证明我们的实验管线（同步参数分组 / 两个优化器 / 记录 loss 与耗时）
在真实训练前是**跑得通且可复现**的。

诚实边界：
  - 数据是合成的，**不是 FineWeb**；loss 是 MSE，**不是 BPB**。
  - 它不能证明「Muon 优于 AdamW」，只能证明「对照实验的代码路径是通的」。
  - 真实 BPB 需要 8xH100，本机无法给出。

修正记录：初版把隐藏层梯度写成 `xb.T @ (d_pred @ W2.T) * (1 - h*h)`，
在 din != batch 时维度不匹配（broadcast 失败）。正确形式是先算
dZ = (d_pred @ W2.T) * (1 - h^2)（形状 批次 x 隐藏），再 gW1 = xb.T @ dZ。
"""
import json
import math
import random
import time

from muon import zeropower_via_newtonschulz5, matmul, transpose, gauss_matrix

SEED = 20261005


def make_data(n=512, din=12, dout=4, seed=SEED):
    rng = random.Random(seed)
    W_true = gauss_matrix(rng, din, dout, scale=1.0 / math.sqrt(din))
    X = gauss_matrix(rng, n, din)
    Y = matmul(X, W_true)
    for row in Y:  # 行归一化到单位长度，避免目标尺度漂移
        nrm = math.sqrt(sum(v * v for v in row)) + 1e-9
        for j in range(len(row)):
            row[j] /= nrm
    return X, Y


def adamw_scalar(w, g, m, v, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mh = m / (1 - b1)
    vh = v / (1 - b2)
    return w - lr * (mh / (math.sqrt(vh) + eps) + wd * w), m, v


def train(optimizer="adamw", steps=120, hidden=24, batch=48, lr=3e-2):
    X, Y = make_data()
    n = len(X)
    din = len(X[0])
    dout = len(Y[0])
    rng = random.Random(SEED + 1)
    W1 = gauss_matrix(rng, din, hidden, scale=1.0 / math.sqrt(din))     # 隐藏层 -> Muon/AdamW
    W2 = gauss_matrix(rng, hidden, dout, scale=1.0 / math.sqrt(hidden))  # 输出层 -> AdamW
    m1 = [[0.0] * hidden for _ in range(din)]
    v1 = [[0.0] * hidden for _ in range(din)]
    m2 = [[0.0] * dout for _ in range(hidden)]
    v2 = [[0.0] * dout for _ in range(hidden)]
    buf1 = [[0.0] * hidden for _ in range(din)]  # Muon 动量缓冲

    t0 = time.time()
    losses = []
    for _ in range(steps):
        idx = [rng.randrange(n) for _ in range(batch)]
        xb = [X[i] for i in idx]
        yb = [Y[i] for i in idx]

        H = matmul(xb, W1)
        for r in H:
            for j in range(len(r)):
                r[j] = math.tanh(r[j])
        pred = matmul(H, W2)
        loss = sum((pred[i][j] - yb[i][j]) ** 2
                   for i in range(batch) for j in range(dout)) / batch
        losses.append(loss)

        d_pred = [[(pred[i][j] - yb[i][j]) / batch for j in range(dout)]
                  for i in range(batch)]
        gW2 = matmul(transpose(H), d_pred)                    # hidden x dout
        dZ = matmul(d_pred, transpose(W2))                    # batch x hidden
        for i in range(batch):
            for j in range(hidden):
                dZ[i][j] *= (1.0 - H[i][j] * H[i][j])         # tanh'
        gW1 = matmul(transpose(xb), dZ)                       # din x hidden

        if optimizer == "adamw":
            for i in range(din):
                for j in range(hidden):
                    W1[i][j], m1[i][j], v1[i][j] = adamw_scalar(
                        W1[i][j], gW1[i][j], m1[i][j], v1[i][j], lr=lr)
        else:  # muon：隐藏层用正交化更新，输出层仍走 AdamW
            for i in range(din):
                for j in range(hidden):
                    buf1[i][j] = 0.95 * buf1[i][j] + gW1[i][j]
            upd = zeropower_via_newtonschulz5(buf1)
            scale = 0.2 * math.sqrt(max(din, hidden))
            for i in range(din):
                for j in range(hidden):
                    W1[i][j] -= lr * scale * upd[i][j]
        for i in range(hidden):
            for j in range(dout):
                W2[i][j], m2[i][j], v2[i][j] = adamw_scalar(
                    W2[i][j], gW2[i][j], m2[i][j], v2[i][j], lr=lr)

    return {
        "optimizer": optimizer,
        "steps": steps,
        "loss_first": round(losses[0], 6),
        "loss_last": round(losses[-1], 6),
        "loss_min": round(min(losses), 6),
        "wallclock_s": round(time.time() - t0, 3),
        "reduced_by_pct": round(100 * (1 - losses[-1] / losses[0]), 2),
    }


def self_test():
    a = train("adamw")
    b = train("muon")
    # 只断言「两条路径都真的在学」——不断言谁更好（那是真实实验的职责）
    assert a["reduced_by_pct"] > 50, a
    assert b["reduced_by_pct"] > 50, b
    assert a["wallclock_s"] > 0 and b["wallclock_s"] > 0
    return {"adamw": a, "muon": b}


if __name__ == "__main__":
    r = self_test()
    print("[smoke] A/B 机制自测 PASS（合成数据 / MSE，非 BPB，非真实成绩）")
    print(json.dumps(r, ensure_ascii=False, indent=2))
