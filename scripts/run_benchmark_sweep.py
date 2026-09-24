from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the A100 decode-attention benchmark matrix")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/sweep.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cases = itertools.product(
        (1, 4, 16),
        (128, 512, 2048),
        (32, 8, 1),
        ("float16", "bfloat16"),
    )
    with args.output.open("w") as output:
        for batch, context, kv_heads, dtype in cases:
            block_size = 16
            blocks_per_request = (context + block_size - 1) // block_size
            command = [
                sys.executable,
                "-m",
                "inferkernellab.benchmark",
                "--device",
                args.device,
                "--backend",
                "auto",
                "--dtype",
                dtype,
                "--batch-size",
                str(batch),
                "--context-len",
                str(context),
                "--num-heads",
                "32",
                "--num-kv-heads",
                str(kv_heads),
                "--head-dim",
                "64",
                "--block-size",
                str(block_size),
                "--num-blocks",
                str(batch * blocks_per_request + blocks_per_request - 1),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
            ]
            record = subprocess.run(command, check=True, capture_output=True, text=True)
            output.write(json.dumps(json.loads(record.stdout), separators=(",", ":")) + "\n")
            output.flush()
            print(
                f"batch={batch} context={context} kv_heads={kv_heads} dtype={dtype}",
                flush=True,
            )


if __name__ == "__main__":
    main()
