from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    configs = (
        (1, 128, 8, 8),
        (4, 128, 8, 2),
        (8, 512, 32, 8),
    )
    for batch, context, heads, kv_heads in configs:
        command = [
            sys.executable,
            "-m",
            "inferkernellab.benchmark",
            "--device",
            "cuda",
            "--backend",
            "auto",
            "--batch-size",
            str(batch),
            "--context-len",
            str(context),
            "--num-heads",
            str(heads),
            "--num-kv-heads",
            str(kv_heads),
            "--head-dim",
            "64",
            "--block-size",
            "16",
            "--num-blocks",
            str(max(256, batch * ((context + 15) // 16))),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(json.loads(result.stdout))


if __name__ == "__main__":
    main()

