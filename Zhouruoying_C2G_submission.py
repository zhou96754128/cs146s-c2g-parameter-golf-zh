"""C2G 提交件打包与硬门禁校验（≤16,000,000 字节）。

用法：
  python3 Zhouruoying_C2G_submission.py pack  --artifact <目录>  --out <tar.gz>
  python3 Zhouruoying_C2G_submission.py verify --file <tar.gz>

诚实边界（写死在产物里，不允许含糊）：
  本机无 8×H100，无法产出真实模型权重。若 artifact 目录不存在或为空，
  本脚本打包一份**明确标注为占位**的最小载荷，并在 submission.json 写入
  {"measured": false, "bpb": null, ...}。
  **绝不伪造 BPB 数字**；真实提交件的 BPB 只能由 8×H100 训练产出。
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import time

ARTIFACT_LIMIT = 16_000_000


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pack(artifact_dir, out_path, meta_path):
    files = []
    if artifact_dir and os.path.isdir(artifact_dir):
        for root, _dirs, names in os.walk(artifact_dir):
            for n in names:
                p = os.path.join(root, n)
                files.append((p, os.path.relpath(p, artifact_dir)))
    placeholder = not files
    if placeholder:
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                               "_placeholder")
        os.makedirs(tmp_dir, exist_ok=True)
        notice = os.path.join(tmp_dir, "PLACEHOLDER.txt")
        with open(notice, "w", encoding="utf-8") as f:
            f.write("本包为占位载荷：本机无 8×H100，未产出真实模型权重。\n"
                    "真实 submission.tar.gz 由 8×H100 训练后重新打包生成。\n")
        files = [(notice, "PLACEHOLDER.txt")]

    with tarfile.open(out_path, "w:gz") as tar:
        for p, arc in files:
            tar.add(p, arcname=arc)

    size = os.path.getsize(out_path)
    meta = {
        "challenge": "C2G",
        "challengeId": "ch-20260717031359-b8wyg0",
        "author": "Zhouruoying",
        "artifact": os.path.basename(out_path),
        "artifact_bytes": size,
        "artifact_limit": ARTIFACT_LIMIT,
        "fits_limit": size <= ARTIFACT_LIMIT,
        "file_count": len(files),
        "placeholder": placeholder,
        "measured": False,
        "bpb": None,
        "baseline_bpb": 1.2244,
        "reason": "本机无 8×H100，未产出真实权重与真实 BPB；占位打包用于证明门禁链路可跑。",
        "sha256": sha256_file(out_path),
        "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0 if meta["fits_limit"] else 1


def verify(path):
    size = os.path.getsize(path)
    ok = size <= ARTIFACT_LIMIT
    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
    print(json.dumps({"file": os.path.basename(path), "bytes": size,
                      "limit": ARTIFACT_LIMIT, "fits": ok,
                      "members": names, "verdict": "PASS" if ok else "FAIL"},
                     ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="C2G 提交件打包/校验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--artifact", default="")
    p.add_argument("--out", default="Zhouruoying_C2G_submission.tar.gz")
    p.add_argument("--meta", default="Zhouruoying_C2G_submission.json")
    v = sub.add_parser("verify")
    v.add_argument("--file", required=True)
    a = ap.parse_args()
    if a.cmd == "pack":
        return pack(a.artifact, a.out, a.meta)
    return verify(a.file)


if __name__ == "__main__":
    sys.exit(main())
