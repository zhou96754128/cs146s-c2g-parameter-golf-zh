# C2G 方案设计 —— Muon 单点优化

**作者**：Zhouruoying　|　**日期**：2026-10-05　|　**版本**：v1
**配套**：`Zhouruoying_C2G_方案草案.md`（方向与证据）、`Zhouruoying_C2G_AI日志.md`、`Zhouruoying_C2G_拿来说明.md`

---

## 1. 目标与验收线

| 项目 | 约定 |
|---|---|
| 硬约束 | 8×H100 SXM / 10 min wallclock / model ≤ 16,000,000 B / 无外网 / 评测 ≤10 min |
| 指标 | FineWeb validation 上的 **BPB**（越低越好） |
| baseline | ≈ 1.2244 BPB（9 层 512 维，tied embedding，4 KV heads） |
| 本轮目标 | 通过 Level 1；**主目标 Level 2：BPB < 1.18**（对应 +$100） |
| 拉伸目标 | Level 3：BPB < 1.12（+$300；按挑战材料为 2026-04 Top 50 门槛） |

> 判定只看**B 组相对 A 组的差值**（同为 3 seed 均值），不看单次最好值。

---

## 2. 系统总览：动什么、不动什么

```
FineWeb 数据管线 ──(不动)──► tokenizer(不动) ──► 9L/512d 模型(不动)
                                                      │
                                        参数分组(动) ──┴──► optimizer
                                                          ├─ 2D 隐藏层权重 → Muon
                                                          └─ embedding / final head / 1D → AdamW
```

**能动**：optimizer 及其配套 lr 分组。
**不能动（保证单变量）**：数据、tokenizer、模型结构、序列长度、batch 大小、评测脚本，全部与 baseline 逐字一致。

---

## 3. 技术选型：Muon 的具体配置

| 项 | 取值 | 依据 |
|---|---|---|
| 适用参数 | 仅隐藏层 2D 权重矩阵 | Muon 原文：标量/向量参数与输入输出层应留给 AdamW |
| 动量 | Nesterov 动量（β≈0.95） | Muon 原文默认 |
| 正交化 | Newton–Schulz 5 步，a=3.4445, b=−4.7750, c=2.0315, eps=1e-7 | Muon 原文给出的系数 |
| 学习率 | 与 AdamW **分组设置**（Muon 组与 AdamW 组各自独立 lr） | 两者尺度不同，不能共用一个 lr |
| embedding / head | **保持 AdamW** | 原文要求，且改动最小 |
| 额外开销 | < 1% FLOPs | 原文 runtime 分析 |

**为什么这样配**：Muon 的收益来自"对 2D 梯度做正交化更新"，只对矩阵型参数成立；对 embedding/head 强行套用会破坏其稀疏/低频更新的特性，反而变差。因此"只换一半"不是妥协，是原文的正确用法。

---

## 4. 超参协议

- **warmup + 余弦退火**：保持 baseline 的 schedule **不变**，以便隔离 optimizer 的影响。
- **lr 搜索**：Muon 组 lr 单独扫 `{0.01, 0.02, 0.04}` 三个值（只在 1 个 seed 上快速定档），AdamW 组沿用 baseline lr。
- **训练时长**：目标 **8:00** 完成训练循环，留 20% buffer 给 10 分钟硬墙。
- **超限保护**：循环末尾自动断言 `artifact_bytes ≤ 16_000_000`，超限直接 fail（不静默通过）。

---

## 5. 实验矩阵

| 编号 | 配置 | seed 数 | 目的 |
|---|---|---|---|
| 001 | baseline（AdamW） | 3 | 建立本地 A 组基线（对齐 1.2244） |
| 002 | **Muon**（本方案主攻） | 3 | 主实验：B−A 差值 |
| 003 | Muon + NS 步数 5→3 | 2 | 时间权衡（仅在 002 超时时启用） |
| 004 | SP8192（预案方向） | 3 | 仅当 002 失败时切换到数据层 |
| 005 | Muon + SP8192 | 3 | 仅当 002 为正时做"叠加"边际验证 |

矩阵按**序号顺序推进**，用前一步结果决定是否进入下一步——不做全矩阵并行烧钱（对应"不 try everything"）。

---

## 6. 对照与统计

- **单变量**：001 与 002 只差 optimizer，其余逐字相同。
- **多 seed**：每组 ≥3 seed，报告 **均值 ± stderr**。
- **判定规则**：
  - 若 `均值(B) + stderr(B) < 均值(A) − stderr(A)` → 判定 Muon **有效**（差值区间不重叠）；
  - 若区间重叠 → 判定为**无定论**，不宣称提升，按 §7 决定是否加 seed 或切换预案。
- **禁止**：不得在 validation set 上调参；不得用单次最好值报成绩。

---

## 7. 边际提升策略

按"每一步只加一个变量、每步都留下可解释的差值"推进：

1. `001 → 002`：拿到 Muon 的边际收益（预期主要增益来源）。
2. 若 002 为正 → `002 → 005`：叠加 SP8192，验证两者是否**可加**（若收益叠加则逼近 <1.12）。
3. 若仍有预算 → 做**消融**：单独关掉 NS 正交化（退化为 Nesterov SGD），证明"收益来自正交化"而非单纯动量。

每一步的差值都写进 `ablation.md`，形成可解释的因果链。

---

## 8. 预算与时间纪律（$25 怎么花）

| 步骤 | 内容 | 花费估算 |
|---|---|---|
| ① 环境与 smoke test | 8×H100 租机 + 跑通 baseline | ~$8 |
| ② 主对照（001/002 各 1 seed） | 拿差值方向 | ~$8 |
| ③ 补 seed 到 3 或切换预案 | 巩固或止损 | ≤$9 |

**止损线**：单个方向消耗达 **$15** 仍无正向差值 → 立即执行预案（§9）。

---

## 9. 风险与回退矩阵

| 风险 | 触发信号 | 动作 |
|---|---|---|
| Muon 不收敛 | loss 发散/NaN | 降低 Muon 组 lr；仍失败 → 切 004（SP8192） |
| Muon 收敛但超时 | wallclock > 9:00 | 减 NS 步数（003），或降 Muon 应用范围 |
| 收益不显著 | 差值区间重叠 | 加 seed；仍重叠 → 判定无效并切预案 |
| artifact 超 16MB | 断言触发 | 启用 int6 embedding 量化（不动 attention） |
| 租不到 8×H100 | 无可用配额 | 延期执行，先交付方案文档（本轮平台要求） |

---

## 10. 复现性设计

- **环境**：`pip install torch>=2.3 sentencepiece numpy tqdm`；`flash-attn`（`--no-build-isolation`）；
  `git clone https://github.com/openai/parameter-golf.git`；`python download_fineweb.py`。
- **固定**：seed（每组显式记录）、数据版本、依赖版本、GPU 型号与数量。
- **日志字段**：`实验编号 / seed / 配置 / BPB / artifact_bytes / wallclock / git_commit`。
- **一键复跑**：每个配置一个 shell 入口；本地可先跑"约束检查 + 指标复算"，机器到位后同一入口直接跑真训练。

---

## 11. 交付物清单（对齐命名规范）

必需 4 项：`Zhouruoying_C2G_方案草案.md` / `_方案设计.md`（本文）/ `_AI日志.md` / `_AAR.md`。
跑通后追加：`_train_gpt.py` / `_submission.tar.gz` / `_submission.json` / `_logs/`（≥3 seed）/ `_ablation.md` / `_leaderboard.md`。
