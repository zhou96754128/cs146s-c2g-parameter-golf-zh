# C2G BPB 对比表（排行榜提交计划）

**作者** Zhouruoying ｜ **挑战** C2G（ch-20260717031359-b8wyg0）
**用途** Level 3 门槛材料之一 + 官方榜单提交计划。

## 1 对比表（方案 vs baseline vs SOTA）
| 方案 | FineWeb val BPB | 状态 | 来源 |
| --- | --- | --- | --- |
| 我们的方案（Muon 单点） | **2.2236**（本机缩比，非 FineWeb 口径） | 本机 CPU / 600 步 / 3 seeds | 本项目《实测记录》 |
| 对照组（AdamW @1e-3） | **3.2061**（同上缩比条件） | 本机 CPU / 600 步 / 3 seeds | 本项目《实测记录》 |
| baseline（官方 starter） | ≈1.2244（≈9:30） | 参考 | CHALLENGE.md [报告值] |
| 当前榜首 | 1.0810（SP8192 + 3-Layer Recurrence） | 参考 | CHALLENGE.md [报告值] |

> FineWeb 那一列仍是空缺且不填：**没有 8×H100 就不声称 FineWeb 榜单数字**。本机缩比实测另列并显式标注「非 FineWeb 口径」，**不与榜首 1.0810 同栏比较**。

## 2 提交路线
1. 依据官方 `openai/parameter-golf` 的 `submission.json` 字段（BPB／训练时间／硬件／seed）打包；
2. 训练循环末尾**自动检查 artifact ≤ 16,000,000 字节**（本机打包门禁已双向验证：196B 放行、17.0MB 拦截 [本机已证]）；
3. 只把**最好且可复现**的方案压成 `submission.tar.gz`，写 README，push GitHub，提交官方榜单（**平台提交已完成**：sub-20261005122533-2afbb7a4；**官方榜单**待算力券跑出 FineWeb 口径成绩后再提交）；
4. 榜单提交前跑一次 `run_all.py` 留存退出码与日志。

## 3 距离目标的差距（诚实版）
- baseline → L2（<1.18）：需约 −0.045；
- L2 → L3（<1.12）：再约 −0.06；
- 榜首 1.0810 证明 1.12 以下可达，但那是 **SP8192 + Recurrence** 的复合配方，非单点；
- 本项目的单点上限未知，**实测前不做承诺**。

## 4 提交记录（滚动更新）
| 日期 | 提交 | BPB | 硬件 | seed | 结果 |
| --- | --- | --- | --- | --- | --- |
| 2026-10-05 | **sub-20261005122533-2afbb7a4** | 2.2236（缩比） | 本机 CPU | 20261005/06/07 | 已提交平台，AI 初评 77/100，待教师终审 |

## 5 复现
```
cd Zhouruoying_C2G_本地校验 && python3 run_all.py   # 8/8 PASS, exit 0
python3 ../Zhouruoying_C2G_submission.py pack <dir> <out.tar.gz> <out.json>
```
