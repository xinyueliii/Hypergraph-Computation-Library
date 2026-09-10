"""Create a deterministic THU-HSEVI-shaped scene for SuperFast smoke tests.

The generated files exercise the upstream loader and checkpoint-backed model,
but they are synthetic and must not be reported as an official dataset result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_scene(output_dir: Path, *, height: int = 64, width: int = 64) -> None:
    if height < 32 or width < 32:
        raise ValueError("SuperFast smoke frames must be at least 32x32 pixels.")

    output_dir = output_dir.resolve()
    frame_dir = output_dir / "frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:height, 0:width]
    for index in range(200):
        frame = ((3 * xx + 5 * yy + 2 * index) % 256).astype(np.uint8)
        square_size = max(4, min(height, width) // 8)
        top = (index * 2) % (height - square_size + 1)
        left = (index * 3) % (width - square_size + 1)
        frame[top : top + square_size, left : left + square_size] = 255
        Image.fromarray(frame, mode="L").save(frame_dir / f"{index}.jpg", quality=95)

    timestamps = np.arange(200, dtype=np.float64)
    (output_dir / "ts_frame.txt").write_text(
        "".join(f"{value:.6f}\n" for value in timestamps),
        encoding="utf-8",
    )

    event_count = 40_000
    event_index = np.arange(event_count, dtype=np.int64)
    event_time = np.linspace(0.01, 198.99, event_count, dtype=np.float64)
    event_x = (7 * event_index + event_index // 97) % width
    event_y = (11 * event_index + event_index // 53) % height
    polarity = event_index % 2
    events = np.column_stack((event_time, event_x, event_y, polarity)).astype(np.float32)
    np.save(output_dir / "event.npy", events)

    (output_dir / "SYNTHETIC_SAMPLE.txt").write_text(
        "This directory is a deterministic synthetic THU-HSEVI-shaped smoke input.\n"
        "It is not an official dataset scene and must not be used for benchmark claims.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    create_scene(args.output_dir, height=args.height, width=args.width)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
