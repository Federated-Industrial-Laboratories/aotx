#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Create and inspect explicit policy ABI versions without executing native code.
# Inputs: policy pack executable. Output: check counts. Exit: 0 pass, 1 failure, 2 usage.
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

checks = 0


def check(ok, text):
    global checks
    checks += 1
    if not ok:
        raise AssertionError(text)


def invoke(exe, args, status=0):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    check(result.returncode == status, f"status {status}: {args}: {result.stderr}")
    return result.stdout


def batch(exe, root, count):
    start = checks
    for row in range(count):
        provenance = root / "provenance.txt"
        provenance.write_text(f"Source {row}; settings {row * 19 + 3}\n")
        for mode in ("supplied", "rules", "native"):
            common = ["--mode", mode, "--minimum-move", row + 1, "--backoff", row + 3,
                      "--provenance", provenance, "--license", root / "LICENSE"]
            if mode == "native":
                common += ["--image", root / "entry.ptx", "--format", "ptx", "--kernel", "aotx_entry",
                           "--architecture", 86, "--state-schema", row + 1, "--state-bytes", 16 + row,
                           "--threads", 64, "--registers", 64, "--shared-bytes", 0, "--local-bytes", 0]
            files = []
            for abi in (None, 1, 2):
                path = root / f"policy-{count}-{row}-{mode}-{abi}.bin"
                text = invoke(exe, ["--output", path, *common, *([] if abi is None else ["--abi", abi])])
                values = dict(line.split("=", 1) for line in text.splitlines())
                data = path.read_bytes()
                check(values["abi"] == str(abi or 1), "inspection reports the exact selected ABI")
                check(int.from_bytes(data[16:20], "little") == (abi or 1), "the bundle stores the exact ABI")
                check(hashlib.sha256(data).hexdigest() == values["policy_digest"], "the digest binds the ABI and every bundle byte")
                check(invoke(exe, ["--inspect", path]) == text, "separate inspection preserves every field")
                files.append(data)
                path.unlink()
            check(files[0] == files[1], "default packing preserves exact ABI 1 bytes")
            check(files[1][:16] == files[2][:16] and files[1][20:] == files[2][20:],
                  "selecting ABI 2 changes only its explicit bundle header field")
    print(f"policy CLI N={count}: {checks - start} checks")


def main():
    if len(sys.argv) != 2:
        return 2
    exe = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="aotx-policy-abi-") as directory:
        root = Path(directory)
        (root / "LICENSE").write_text("Apache-2.0\n")
        (root / "entry.ptx").write_text(
            ".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry aotx_entry() { ret; }\n")
        for count in (1, 64):
            batch(exe, root, count)
        path = root / "refused.bin"
        common = ["--output", path, "--mode", "rules", "--provenance", root / "provenance.txt",
                  "--license", root / "LICENSE"]
        for abi in (0, 3, "x", 4294967296):
            invoke(exe, [*common, "--abi", abi], 1)
            check(not path.exists(), "unsupported ABI leaves no output file")
        invoke(exe, [*common, "--abi", 1, "--abi", 2], 2)
        check(not path.exists(), "duplicate ABI selection leaves no output file")
    print(f"policy CLI: {checks} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
