# C2G BPB 对比表（排行榜提交计划）

**作者** Zhouruoying ｜ **挑战** C2G（ch-20260717031359-b8wyg0）
**用途** Level 3 门槛材料之一 + 官方榜单提交计划。

## 1 对比表（方案 vs baseline vs SOTA）
| 方案 | FineWeb val BPB | 状态 | 来源 |
| --- | --- | --- | --- |
| 我们的方案（Muon 单点） | [待实测] | 待 8×H100 | 本项目 |
| baseline（官方 starter） | ≈1.2244（≈9:30） | 参考 | CHALLENGE.md [报告值] |
| 当前榜首 | 1.0810（SP8192 + 3-Layer Recurrence） | 参考 | CHALLENGE.md [报告值] |

> 空缺不是遗漏：**没有 H100 就不填任何 BPB 数字**。本表拒绝用本地缩比数字冒充。

## 2 提交路线
1. 依据官方 `openai/parameter-golf` 的 `submission.json` 字段（BPB／训练时间／硬件／seed）打包；
2. 训练循环末尾**自动检查 artifact ≤ 16,000,000 字节**（本机打包门禁已双向验证：196B 放行、17.0MB 拦截 [本机已证]）；
3. 只把**最好且可复现**的方案压成 `submission.tar.gz`，写 README，push GitHub，提交官方榜单；
4. 榜单提交前跑一次 `run_all.py` 留存退出码与日志。

## 3 距离目标的差距（诚实版）
- baseline → L2（<1.18）：需约 −0.045；
- L2 → L3（<1.12）：再约 −0.06；
- 榜首 1.0810 证明 1.12 以下可达，但那是 **SP8192 + Recurrence** 的复合配方，非单点；
- 本项目的单点上限未知，**实测前不做承诺**。

## 4 提交记录（滚动更新）
| 日期 | 提交 | BPB | 硬件 | seed | 结果 |
| --- | --- | --- | --- | --- | --- |
| — | 尚未提交（无 H100） | — | — | — | 待算力券到位 |

## 5 复现
```
cd Zhouruoying_C2G_本地校验 && python3 run_all.py   # 7/7 PASS, exit 0
python3 ../Zhouruoying_C2G_submission.py pack <dir> <out.tar.gz> <out.json>
```
