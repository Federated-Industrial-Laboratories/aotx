#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check memory configuration through the actual build system and shared headers.
# Inputs: source directory, CMake and CUDA compiler paths. Output: check counts.
# Exit: 0 pass, 1 failed check, 4 bad arguments.
import pathlib
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) != 4:
        return 4
    source, cmake, cuda = sys.argv[1:]
    checks = 0
    with tempfile.TemporaryDirectory(prefix="aotx-memory-config-") as directory:
        base = [cmake, "-S", source, "-B", directory, "-G", "Ninja",
                "-DCMAKE_CUDA_COMPILER=" + cuda]
        for objects, payload in (("0", "1"), ("-1", "1"), ("abc", "1"),
                                 ("01", "1"), ("1", "0"), ("1", "-2"),
                                 ("8388608", "1"), ("1", "2147483647"),
                                 ("999999999999999999999", "1")):
            result = subprocess.run(base + ["-DAOTX_MEMORY_OBJECTS=" + objects,
                "-DAOTX_MEMORY_BYTES=" + payload], capture_output=True, text=True)
            output = result.stdout + result.stderr
            if result.returncode == 0 or "transfer" not in output:
                print(output); return 1
            checks += 1
        result = subprocess.run(base + ["-DAOTX_MEMORY_OBJECTS=8193", "-DAOTX_MEMORY_BYTES=16777217"],
                                capture_output=True, text=True)
        if result.returncode:
            print(result.stdout + result.stderr); return 1
        checks += 1
        cache = (pathlib.Path(directory) / "CMakeCache.txt").read_text()
        if "AOTX_MEMORY_OBJECTS:STRING=8193" not in cache or "AOTX_MEMORY_BYTES:STRING=16777217" not in cache:
            return 1
        checks += 1
        for cap, valid in (("0", True), ("34359738368", True), ("-1", False), ("123", False), ("bad", False),
                           ("9223372036854775808", False), ("999999999999999999999", False)):
            result = subprocess.run(base + ["-DAOTX_CCIR_FILE_BYTES=" + cap], capture_output=True, text=True)
            if (result.returncode == 0) != valid:
                print(result.stdout + result.stderr); return 1
            checks += 1
        probe = pathlib.Path(directory) / "bounds.cu"
        probe.write_text('#include "cognitive/live.cuh"\n'
            'static_assert(AOTX_COG_WORDS * 32 >= AOTX_COG_OBJECTS, "last dependency word");\n'
            'static_assert((AOTX_COG_WORDS - 1) * 32 < AOTX_COG_OBJECTS, "word count");\n'
            'static_assert(AOTX_LIVE_BYTES >= AOTX_LIVE_TEXT_CHOICES, "input batch bytes");\n'
            'static_assert(AOTX_LIVE_RESULTS >= AOTX_LIVE_TEXT_CHOICES, "output batch bytes");\n')
        for objects, payload, valid in (("1", "1", True), ("8193", "16777217", True),
                                        ("8388608", "1", False),
                                        ("72057594037927936ULL", "1", False),
                                        ("1", "18446744073709551232ULL", False)):
            result = subprocess.run([cuda, "-std=c++17", "-c", str(probe), "-o", directory + "/bounds.o",
                "-I" + source + "/cuda", "-DAOTX_MEMORY_OBJECTS=" + objects,
                "-DAOTX_MEMORY_BYTES=" + payload], capture_output=True, text=True)
            if (result.returncode == 0) != valid or (not valid and "transfer byte count" not in result.stderr):
                print(result.stdout + result.stderr); return 1
            checks += 1
    print(f"memory configuration: {checks} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
