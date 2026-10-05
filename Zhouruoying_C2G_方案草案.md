# C2G 参数高尔夫 —— 方案草案

**作者**：Zhouruoying　|　**日期**：2026-10-05　|　**目标 Level**：Level 1（申请 $25 算力券），冲刺 Level 2（BPB < 1.18）
**挑战**：C2G / ch-20260717031359-b8wyg0 —— 极限约束下的语言模型训练

---

## 0. 一句话方向

把训练的优化器从 **AdamW** 换成 **Muon（Nesterov 动量 + Newton–Schulz 正交化）**，只作用于**隐藏层的 2D 权重**；embedding 与输出 head 仍用 AdamW。除了 optimizer，其余**一切不动**。

---

## 1. 为什么是"这一个"方向，而不是"都试一遍"

C2G 的硬约束是**固定的 10 分钟 wallclock + ≤16MB artifact**。所以真正的问题不是"选哪个技巧"，而是：

> **在同样的 10 分钟里，谁能让模型学到更多？**

四类方向里——

| 方向 | 作用层面 | 与"限时"这条主链的关系 |
|---|---|---|
| tokenizer（SP8192） | 数据表示 | 间接：改变序列长度/压缩率，需要重做数据管线 |
| 架构（recurrence / attention 变体） | 模型结构 | 间接：改动大、与前两者可能耦合 |
| 量化（int6） | 存储 | 间接：主要影响能否塞进 16MB，不直接提升学习效率 |
| **optimizer（Muon）** | **单位算力 → 学习量** | **直接命中"限时"这一最紧约束** |

因此选定**单点方向 = 优化器**。两个理由：

1. **它最直接命中最紧约束**：10 分钟是硬墙，优化器决定的恰好是"这 10 分钟能换来多少有效学习"。
2. **它最容易做成干净的单变量对照**：除 optimizer 外（数据、模型、评测流程）完全一致，满足 rubric 中"有对照实验、策略可解释"的要求。

其余方向**不在第一轮并行铺开**，仅作为 §4 的失败预案保留——这正是"不 try everything"与"预留退路"的平衡。

---

## 2. 证据（真实、可核验）

**证据 1（论文 / 原始出处）**：Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein.
*Muon: An optimizer for hidden layers in neural networks*, 2024-12-08.
https://kellerjordan.github.io/posts/muon/ 　（本机已**直接抓取原文核验**，非转述）

原文给出的关键结论：

- Muon 相比 AdamW 在 **NanoGPT speedrun** 上把训练速度提升 **1.35×（约 35%）**；
- 此后**连续 12 次** NanoGPT 训练记录均由 Muon 保持，来自 **7 位不同研究者**——这说明它不是单次运气；
- Muon 的额外 FLOP 开销 **< 1%**（原文 runtime 分析）。在"限时"场景下，这个开销几乎可以忽略，等于"白拿"的样本效率。

**证据 2（历史提交记录）**：https://github.com/KellerJordan/modded-nanogpt
Muon 即在这个竞技仓库的 speedrun 中被提出并验证，是其官方引用来源。挑战材料 CHALLENGE.md 亦把 Muon 列为"当前 Parameter Golf **榜首的优化器**"。

**证据 3（改动面极小、可回退）**：Muon 只作用于 2D 隐藏层权重，**embedding 与 final head 必须继续用 AdamW**（Muon 原文 empirical 部分明确要求）。因此本方案的改动范围只有 optimizer 一处。

> **诚实边界**：SP8192 的 −0.03 ~ −0.05 BPB、以及当前榜首 1.0810 BPB（SP8192 + 3-Layer Recurrence）这两组数字，来自**挑战材料 CHALLENGE.md 的陈述**，我**未在 H100 上独立复现**；它们仅作为方向背景引用，**不作为本方案的证据主张**。

---

## 3. 实验计划与 $25 怎么花

**实验设计（单变量对照）**

- A 组 = baseline（AdamW），B 组 = Muon + AdamW(head/emb)，**其余超参逐字一致**；
- 每个配置跑 **3 个 seed**，报告 BPB 的**均值 ± stderr**——避免把单次噪声误判成"提升"（挑战"血的教训"第 1 条）；
- 评测固定在 FineWeb validation 上，用 BPB（越低越好），不看 perplexity。

**预算节奏（=$25 怎么花）**

| 阶段 | 内容 | 预估花费 |
|---|---|---|
| 第 0 步 | 租 8×H100 SXM，跑通 baseline，确认 BPB ≈ 1.2244、耗时 ≈ 9:30 | 约 $25 中的第一笔 |
| 第 1 步 | A/B 各 1 seed 的 smoke test，先拿到 **B−A 的差值方向** | 优先消耗券 |
| 第 2 步 | 差值方向明确后，才补足 3 seed 正式对照 | 余量 |

- 目标时间设为 **8 分钟**，预留 20% buffer（挑战"血的教训"第 3 条）；
- 训练循环末尾**自动断言 artifact ≤ 16,000,000 字节**，超限直接报错（第 2 条）；
- 每次实验落 `logs/`：seed、BPB、artifact 体积、wallclock，全部可复跑。

---

## 4. 失败预案

- **预案 1（主）**：若 Muon 在"限时小模型"设置下**不收敛**，或 **B−A 差值 ≤ 0**（收益被抵消），**回退到 tokenizer 方向**：升级到 SP8192（只改数据预处理、不动模型），按挑战材料预期约 −0.03 ~ −0.05 BPB，风险最低。
- **预案 2**：若 Muon 收敛但**超时**，减少 Newton–Schulz 迭代步数（5 → 3），或只对 Q/K/V 分别应用，观察 BPB 与时间的权衡。
- **止损规则**：单个方向**最多消耗 $25 券的 60%（约 $15）仍无正向差值**，立即切换预案，绝不在一个方向无限加注。

---

## 5. 交付物（对齐命名规范）

平台必交 4 项：

- `Zhouruoying_C2G_方案草案.md`（本文，≥500 字，回答 4 个门槛问题 ✅）
- `Zhouruoying_C2G_方案设计.md`
- `Zhouruoying_C2G_AI日志.md`
- `Zhouruoying_C2G_AAR.md`

跑通后追加：`Zhouruoying_C2G_train_gpt.py`、`Zhouruoying_C2G_submission.tar.gz` / `.json`、`Zhouruoying_C2G_logs/`（≥3 seed）、`Zhouruoying_C2G_ablation.md`、`Zhouruoying_C2G_leaderboard.md`。

---

*四个门槛问题自查：① 单点方向=优化器（§1）✅　② 证据=Muon 原文 + modded-nanoGPT（§2）✅　③ 实验计划与 $25 分配（§3）✅　④ 失败预案（§4）✅*
