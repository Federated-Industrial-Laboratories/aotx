#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# size_gate_test.py: the host glue boundary check.
# Input: The path of the size gate.
# Output: One summary line.
# Exit codes: 0 passed, 1 failed.

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    gate = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="aotx-size-") as root:
        fixture = Path(root) / "boundary_host.cu"
        fixture.write_text("/* fixture */\n" * 300, encoding="ascii")
        run = subprocess.run([sys.executable, str(gate), str(fixture)],
                             capture_output=True, text=True)
        output = run.stdout + run.stderr
        if run.returncode != 1 or "300 lines (limit 300)" not in output:
            print(output, end="")
            print(f"size gate test: expected status 1, got {run.returncode}")
            return 1
    print("size gate test: 300-line host fixture refused with status 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
