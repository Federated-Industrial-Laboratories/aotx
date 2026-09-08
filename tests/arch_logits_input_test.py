# SPDX-License-Identifier: Apache-2.0
"""Check that malformed capture requests fail before a device context opens.

Inputs: the capture executable path.
Outputs: case counts and refusals for one and 64 sequences.
Exit codes: zero when every input is refused, one on failure.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def request(sequences):
    words = [256, 1, sequences, 1] + [1] * sequences + list(range(sequences))
    return bytearray(b"AOTXAC01" + struct.pack("<" + "I" * len(words), *words))


def word(data, index, value):
    result = data.copy()
    struct.pack_into("<I", result, 8 + index * 4, value)
    return result


def main():
    if len(sys.argv) != 2:
        return 1
    cases = failures = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for sequences in (1, 64):
            data = request(sequences)
            variants = {
                "magic": b"BADMAGIC" + data[8:], "empty": b"", "truncated": data[:-1],
                "trailing": data + b"x", "vocab": word(data, 0, 0),
                "no_groups": word(data, 1, 0), "too_many_groups": word(data, 1, 1025),
                "no_sequences": word(data, 2, 0), "too_many_sequences": word(data, 2, 100000),
                "no_steps": word(data, 3, 0), "too_many_steps": word(data, 3, 65),
                "no_prefill": word(data, 4, 0), "long_prefill": word(data, 4, 100000),
                "invalid_id": word(data, 4 + sequences, 256),
            }
            for name, value in variants.items():
                path = root / "input.bin"
                path.write_bytes(value)
                result = subprocess.run([sys.argv[1], str(root / "no-store"), "language",
                                         str(path), str(root / "no-output")], capture_output=True, timeout=10)
                ok = result.returncode == 1 and b"invalid role or input" in result.stderr
                ok &= not (root / "no-output").exists()
                cases += 1
                failures += not ok
                print(f"arch_logits_input: N={sequences} {name}: {'ok' if ok else 'FAILED'}")
    print(f"arch_logits_input: {cases} cases, {failures} failures, 0 skips")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
