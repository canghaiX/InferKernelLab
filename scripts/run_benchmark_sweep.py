from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/sweep.jsonl"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = itertools.product((1, 4, 16), (128, 512, 2048), (8, 2))
    with args.output.open("w") as output:
        for batch, context, kv_heads in cases:
            command = [
                sys.executable, "-m", "inferkernellab.benchmark",
                "--device", args.device, "--backend", "auto",
                "--batch-size", str(batch), "--context-len", str(context),
                "--num-heads", "8", "--num-kv-heads", str(kv_heads),
                "--head-dim", "64", "--block-size", "16",
                "--num-blocks", str(max(256, batch * ((context + 15) // 16))),
                "--warmup", "10", "--iterations", "50",
            ]
            record = subprocess.run(command, check=True, capture_output=True, text=True)
            output.write(json.dumps(json.loads(record.stdout)) + "\n")
            output.flush()
            print(f"batch={batch} context={context} kv_heads={kv_heads}")


if __name__ == "__main__":
    main()

