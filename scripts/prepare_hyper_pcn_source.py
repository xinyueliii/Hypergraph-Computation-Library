"""Apply the minimal reproducible RTX 3090 build compatibility patch to Hyper-PCN."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
NEW = 'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")'
FILES = (
    Path("extensions/chamfer_dist/setup.py"),
    Path("extensions/pointnet2_ops_lib/setup.py"),
)


def patch_source(source_root: Path) -> None:
    for relative_path in FILES:
        path = source_root / relative_path
        text = path.read_text(encoding="utf-8")
        if NEW in text:
            print(f"already patched: {relative_path}")
            continue
        if OLD not in text:
            raise RuntimeError(f"Pinned build line not found in {path}; audit the new source revision.")
        path.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print(f"patched: {relative_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    patch_source(args.source_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
