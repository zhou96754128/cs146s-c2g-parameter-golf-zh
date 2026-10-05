# C2G 参数高尔夫 —— Muon 优化器驱动的极限约束训练方案

**挑战：** C2G 参数高尔夫 —— 极限约束下的语言模型训练
**挑战 ID：** ch-20260717031359-b8wyg0
**姓名拼音：** Zhouruoying
**提交身份：** 2025105400318
**日期：** 2026-10-05

极限约束：8×H100 SXM、训练 10 分钟挂钟、模型 ≤16 MB（16,000,000 字节）、FineWeb 验证集评测（禁止联网下载）、指标 BPB（越低越好）、评测 ≤10 分钟、禁止外部调用。

---

## 一、交付物清单

| 文件 | 对应必交物 | 说明 |
| --- | --- | --- |
| `Zhouruoying_C2G_方案草案.md` | 方案草案 | 单点方向 + 证据 + 实验预算 + 失败预案（Level 1 门槛，≥500 字） |
| `Zhouruoying_C2G_方案设计.md` | 方案设计 | 目标 Level、技术选型、实验矩阵、对照设计与边际提升策略 |
| `Zhouruoying_C2G_AI日志.md` | AI日志 | 本次协作的真实迭代过程与纠偏记录 |
| `Zhouruoying_C2G_拿来说明.md` | 支撑材料 | 引用来源与借用边界逐条核验表 |
| `Zhouruoying_C2G_AAR.md` | AAR | 七维复盘，含失败经验与改进方案 |
| `Zhouruoying_C2G_项目自评.md` | 平台附交 | 按 rubric 五维自评（与 AAR 分属不同文本、不同来源） |
| `Zhouruoying_C2G_本地校验/` | 可复现证据 | 纯标准库校验套件 + 日志 + 退出码 |
| `Zhouruoying_C2G_实测记录.md` / `.json` | 实测记录 | 本机缩比真实训练对照（AdamW vs Muon，3 seeds）的 BPB / 统计 / 硬件 / 耗时 / 退出码 |
| `Zhouruoying_C2G_submission.tar.gz` / `.json` | 提交件 | 打包门禁放行的提交件（含实测字段与口径标注） |
| `Zhouruoying_C2G_AI佐证.md` | 支撑材料 | AI 指令原文、输出片段、采纳-修改-驳回判定、被驳建议的复现验证 |
| `Zhouruoying_C2G_单点改进RFC.md` / `_组合优化RFC.md` / `_ablation.md` / `_leaderboard.md` | 支撑材料 | 单点与组合方案候选、消融表、榜单对比与提交计划 |

## 二、本地可复现校验

本机**无 8×H100**，因此本机跑出的是**缩小规模的真实受控对照**（本机 CPU、600 步、3 seeds、wiki 文本），**不是 FineWeb 口径的榜单成绩**：它给出 Muon 相对 AdamW 的**方向性 BPB 证据**（Δ=0.9825），成绩达成维度按「方案 + 可复现工程验证 + 缩比真实实测」三线并计，并明确标注不可与榜单数值等价。

```bash
cd Zhouruoying_C2G_本地校验
python3 run_all.py
```

约定：`run_all.py` 返回 `0` 表示全部 PASS，返回 `1` 表示存在失败项。

最近一次运行结果：**8/8 PASS，退出码 0**（日志见 `logs/run_all.log`，结构化结果见 `logs/summary.json`）。

覆盖范围（8 项）：

- `bpb.py` —— BPB 计算口径自检（均匀分布、单峰分布、多字节归一、分词器不变性）
- `muon.py` —— Muon 正交化步骤自检（Gram 缺陷 81.972 → 0.4343、对角元范围、形状保持、确定性、零输入安全）
- `budget.py` —— 16 MB 模型预算与 10 分钟训练预算核算
- `smoke_ab.py` —— AdamW / Muon 小规模 A/B 冒烟对照
- `seeds.py` —— 3 seed（20261005 / 20261006 / 20261007）复跑一致性 + 对照统计，落盘 `../Zhouruoying_C2G_logs/`
- `significance.py` —— L3 显著性门禁（精确双侧符号翻转检验、t 临界值手算校验、最小可判 seed 数）
- `package.py` —— 提交件打包门禁（打包件字节数 ≤16,000,000 硬拦截 + 超限负向用例）
- `entry.py` / `Zhouruoying_C2G_train_gpt.py` —— 训练入口自检（`--check` 退出码 0；`--train` 无 torch 时以退出码 3 明确阻塞，不静默失败）

> 诚实边界：本机无 8×H100。除上面的「机制与工程」证据外，本机另有一组**缩小规模真实训练对照**（见 `Zhouruoying_C2G_实测记录.md`：AdamW **3.2061** → Muon **2.2236**，Δ=**0.9825**，3/3 seed 同号，Welch t=57.79）。该数字运行于**本机 CPU / 600 步 / wiki 文本**，与 FineWeb 榜单口径（榜首 1.0810）**不可直接比较**；FineWeb 口径 BPB 仍待 8×H100 复跑回填。

## 三、技术路线一句话

把单点押注放在**优化器**上：用 Muon（MomentUm Orthogonalized by Newton–Schulz，仅作用于 2D 隐层权重，embedding / 最终头 / 1D 参数仍用 AdamW），换取同等步数下更低的 BPB，从而在 10 分钟挂钟与 16 MB 预算内逼近 Levels 门槛。
