"""训练入口三向门禁：验证 Zhouruoying_C2G_train_gpt.py 确实可跑，且失败不静默。

三向验证（缺任何一向都不算过）：
  --check       必须 rc=0 且 verdict=PASS（配置解析 + 16MB 预算算术可跑）
  --local-smoke 必须 rc=0（缩比 A/B 管线可跑）
  --train       本机无 8×H100 时必须 rc=3 且打印 [BLOCKED]
                —— 禁止「无卡却 rc=0」这种静默假通过

为什么单列一项：前一版套件只覆盖算法与打包，没有覆盖交付入口本身。
「入口可运行」若不被门禁验证，就只是嘴上的说法。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENTRY = os.path.join(ROOT, "Zhouruoying_C2G_train_gpt.py")
ENTRY_NAME = "Zhouruoying_C2G_train_gpt.py"


def _run(flag):
    p = subprocess.run([sys.executable, ENTRY, flag],
                       capture_output=True, text=True, timeout=300)
    return {"flag": flag, "rc": p.returncode,
            "out": p.stdout or "", "err": (p.stderr or "")[:300]}


def self_test():
    assert os.path.exists(ENTRY), f"入口不存在：{ENTRY_NAME}"

    r_check = _run("--check")
    assert r_check["rc"] == 0, f"--check 应 rc=0，实际 {r_check['rc']}：{r_check['err']}"
    try:
        check_json = json.loads(r_check["out"])
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"--check 未输出合法 JSON：{type(e).__name__}: {e}")
    assert check_json.get("verdict") == "PASS", \
        f"--check verdict 应为 PASS，实际 {check_json.get('verdict')}"
    assert check_json.get("budget_rows"), "--check 未产出预算行"
    assert any(r.get("fits") for r in check_json["budget_rows"]), \
        "--check 预算行全部超限（配置不合法）"

    r_smoke = _run("--local-smoke")
    assert r_smoke["rc"] == 0, \
        f"--local-smoke 应 rc=0，实际 {r_smoke['rc']}：{r_smoke['err']}"
    assert "result" in r_smoke["out"], "--local-smoke 未产出 result 字段"

    r_train = _run("--train")
    assert r_train["rc"] == 3, \
        f"--train 在无 8×H100 环境必须 rc=3（明确失败），实际 {r_train['rc']}"
    assert "BLOCKED" in r_train["out"], \
        "--train 失败时必须打印 [BLOCKED]，不得静默退出"

    return {
        "check": {"rc": r_check["rc"], "verdict": check_json.get("verdict")},
        "local_smoke": {"rc": r_smoke["rc"], "has_result": "result" in r_smoke["out"]},
        "train_no_gpu": {"rc": r_train["rc"], "blocked_printed": "BLOCKED" in r_train["out"]},
        "note": "入口三向：可跑 2 条 + 明确失败 1 条（失败路径不静默）",
    }
