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

## 二、本地可复现校验

本机**无 H100**，因此**不包含任何真实 BPB 成绩**：本轮交付的是方案与可复现的工程验证，成绩达成维度以方案与可行性证据计。

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

> 诚实边界：本机无 8×H100，以上全部为**机制与工程可复现性**证据，**不含任何真实 BPB 成绩**；BPB 数值一律标注为「预期值」。

## 三、技术路线一句话

把单点押注放在**优化器**上：用 Muon（MomentUm Orthogonalized by Newton–Schulz，仅作用于 2D 隐层权重，embedding / 最终头 / 1D 参数仍用 AdamW），换取同等步数下更低的 BPB，从而在 10 分钟挂钟与 16 MB 预算内逼近 Levels 门槛。
