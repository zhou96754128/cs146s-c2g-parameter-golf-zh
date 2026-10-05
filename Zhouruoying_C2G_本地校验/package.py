"""提交件打包门禁的本地自检 —— 证明「≤16,000,000 字节」这条硬门禁真的会拦人。

做法：在临时目录里造两个载荷——
  A) 合规载荷（小文件）          -> 期望 PASS
  B) 超限载荷（>16MB 不可压缩）  -> 期望 FAIL
两者都通过后，才说明 package/submission 这条链路是**有效的门禁**，
而不是一个永远说 PASS 的摆设。

不会在仓库里留下任何伪造的提交件：全部在系统临时目录内创建并删除。
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIMIT = 16_000_000


def _load_submission_module():
    path = os.path.join(ROOT, "Zhouruoying_C2G_submission.py")
    spec = importlib.util.spec_from_file_location("c2g_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def self_test():
    sub = _load_submission_module()
    tmp = tempfile.mkdtemp(prefix="c2g_pack_")
    try:
        # --- A) 合规载荷 ---
        ok_dir = os.path.join(tmp, "ok")
        os.makedirs(ok_dir)
        with open(os.path.join(ok_dir, "weights.bin"), "wb") as f:
            f.write(b"\x00" * 1024)
        out_ok = os.path.join(tmp, "ok.tar.gz")
        meta_ok = os.path.join(tmp, "ok.json")
        rc_ok = sub.pack(ok_dir, out_ok, meta_ok)
        with open(meta_ok, encoding="utf-8") as f:
            m_ok = json.load(f)
        assert rc_ok == 0 and m_ok["fits_limit"] is True, m_ok
        assert m_ok["measured"] is False and m_ok["bpb"] is None, m_ok

        # --- B) 超限载荷（>16MB，且必须**不可压缩**） ---
        # 门禁测的是打包后 tar.gz 的**实际字节数**；若样本可压缩（稀疏/全零），
        # gzip 会把它压到几 KB，超限就测不出来。这正是本自检第一版抓到的真缺陷。
        big_dir = os.path.join(tmp, "big")
        os.makedirs(big_dir)
        big = os.path.join(big_dir, "weights.bin")
        remaining = LIMIT + 1_000_000
        with open(big, "wb") as f:
            while remaining > 0:
                n = min(1 << 20, remaining)
                f.write(os.urandom(n))
                remaining -= n
        out_big = os.path.join(tmp, "big.tar.gz")
        meta_big = os.path.join(tmp, "big.json")
        rc_big = sub.pack(big_dir, out_big, meta_big)
        with open(meta_big, encoding="utf-8") as f:
            m_big = json.load(f)
        assert rc_big == 1 and m_big["fits_limit"] is False, m_big
        assert m_big["artifact_bytes"] > LIMIT, m_big

        # --- C) verify 子命令必须与 pack 结论一致 ---
        assert sub.verify(out_ok) == 0
        assert sub.verify(out_big) == 1

        return {
            "compliant_payload": {"bytes": m_ok["artifact_bytes"], "fits": True,
                                  "rc": rc_ok, "measured": m_ok["measured"]},
            "oversize_payload": {"bytes": m_big["artifact_bytes"], "fits": False,
                                 "rc": rc_big},
            "verify_consistent": True,
            "note": "两个方向都被验证：合规放行、超限拦截。门禁不是摆设。",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    r = self_test()
    print("[package] self-test PASS —— 16MB 门禁双向验证通过")
    print(json.dumps(r, ensure_ascii=False, indent=2))
