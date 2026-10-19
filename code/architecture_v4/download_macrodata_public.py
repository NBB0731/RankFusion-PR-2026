"""Download the public OddBench NPZ files from the official HF dataset tree."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "macrodata_confirmatory"
DATA = ROOT / "public"
API = "https://huggingface.co/api/datasets/MacrOData-CMU/OddBench/tree/main/public?recursive=true"
BASE = "https://huggingface.co/datasets/MacrOData-CMU/OddBench/resolve/main/"

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(API, headers={"User-Agent": "OddBench-confirmatory-study/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        entries = json.load(response)
    entries = [x for x in entries if x.get("type") == "file" and x.get("path", "").endswith(".npz")]
    (ROOT / "source_file_listing.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Listed {len(entries)} NPZ files; reported total {sum(x.get('size', 0) for x in entries):,} bytes", flush=True)

    def fetch(entry):
        rel = entry["path"]
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == entry.get("size"):
            return rel, dst.stat().st_size, "cached"
        url = BASE + urllib.parse.quote(rel, safe="/")
        req = urllib.request.Request(url, headers={"User-Agent": "OddBench-confirmatory-study/1.0"})
        tmp = dst.with_suffix(dst.suffix + ".partial")
        with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        os.replace(tmp, dst)
        if dst.stat().st_size != entry.get("size"):
            raise IOError(f"Size mismatch for {rel}: {dst.stat().st_size} != {entry.get('size')}")
        return rel, dst.stat().st_size, "downloaded"

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(fetch, entries), 1):
            results.append(result)
            if i % 25 == 0 or i == len(entries):
                print(f"Completed {i}/{len(entries)} files ({sum(r[1] for r in results):,} bytes)", flush=True)
    (ROOT / "download_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Public OddBench download complete", flush=True)

if __name__ == "__main__":
    import urllib.parse
    main()
