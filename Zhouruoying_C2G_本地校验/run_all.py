"""一键入口：跑本地全部校验，输出可复现证据与退出码。

用法：  python3 run_all.py
退出码：0 = 全部 PASS；1 = 有失败
产出：  logs/run_all.log（同时打印到终端）、logs/summary.json
依赖：  仅 Python 3 标准库（无需 numpy / 第三方包，无需外网）

鲁棒性：模块导入也放在 try 内，日志在最后统一落盘；
因此即使导入期就异常（例如缺依赖），本文件仍会写出 logs/run_all.log，
不会「无日志直接退出」。
"""
import io
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

_buf = io.StringIO()


def log(s=""):
    print(s)
    _buf.write(s + "\n")


def main():
    log("=" * 68)
    log("C2G 本地可复现校验（CPU / 纯标准库 / 无外网）")
    log("诚实边界：本机无 8×H100；套件内的 BPB 均为本机缩比真实训练（CPU）")
    log("           实测值，不可与 FineWeb 榜单口径（榜首 1.0810）比较。")
    log("=" * 68)

    results = {}
    failed = 0
    cases = []
    try:
        sys.path.insert(0, HERE)
        import bpb
        import muon
        import budget
        import smoke_ab
        import seeds
        import significance
        import package
        import entry

        cases = [
            ("BPB 指标口径", bpb.self_test),
            ("Muon 正交化", muon.self_test),
            ("16MB 参数预算", budget.self_test),
            ("A/B 机制 smoke", smoke_ab.self_test),
            ("3-seed 可复现 + 对照统计", seeds.self_test),
            ("L3 显著性门禁", significance.self_test),
            ("提交件打包门禁（含超限拦截）", package.self_test),
            ("训练入口三向门禁", entry.self_test),
        ]
    except Exception as e:  # noqa: BLE001
        failed += 1
        log(f"[FAIL] 模块导入 -> {type(e).__name__}: {e}")
        log(traceback.format_exc())

    for name, fn in cases:
        t0 = time.time()
        try:
            out = fn()
            results[name] = out
            log(f"[PASS] {name}  ({time.time() - t0:.2f}s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            log(f"[FAIL] {name}  -> {type(e).__name__}: {e}")
            log(traceback.format_exc())

    log("-" * 68)
    if cases:
        log(f"汇总：{len(cases) - failed}/{len(cases)} PASS")
    else:
        log("汇总：0/0 —— 用例未能加载（见上方导入错误）")

    for name, out in results.items():
        log(f"  * {name}: {json.dumps(out, ensure_ascii=False)[:400]}")

    log("=" * 68)
    if failed == 0:
        log("结论：评测口径 / 正交化实现 / 尺寸预算 / A-B 管线 / 3-seed 复跑 /")
        log("      显著性门禁 / 打包门禁 / 训练入口三向门禁，八项在本机可复现")
        log("      （含三条会真拦人的负向用例：超限载荷 17.0MB / 欠功效显著性 / 无卡训练 rc=3）。")
        log("待机器到位后，同一入口即可替换为真实 8xH100 训练。")
    else:
        log(f"结论：有 {failed} 项未通过，本机校验未达成可复现（详见上方 FAIL）。")
    log("=" * 68)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "run_all.log"), "w", encoding="utf-8") as f:
        f.write(_buf.getvalue())
    with open(os.path.join(LOG_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"exit_code={1 if failed else 0}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
