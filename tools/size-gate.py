#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# size-gate.py: the file size gate.
#
# The gate refuses a file that has more than 1000 lines. It refuses a host glue file
# (a name that ends in _host.cu) that has more than 300 lines. It warns at 800 lines.
#   size-gate.py PATH [PATH...]   examine the given files or directories.
#   size-gate.py --staged         examine staged content only.
#   size-gate.py                  examine all git-tracked files under the current directory.
# Exit codes: 0 clean, 1 findings, 2 usage or environment error.

import subprocess
import sys
from pathlib import Path

CEILING = 1000
HOST_CEILING = 300
WARNING = 800
# Binary content has no lines. A line count of a byte run counts newline bytes, which
# says nothing about the size of the file.
SKIP_SUFFIXES = {".png", ".jpg", ".gif", ".pdf", ".onnx", ".gguf", ".bin", ".zip", ".gz",
                 ".f32"}


def git_files(base, staged):
    cmd = ["git", "-C", str(base)]
    cmd += ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged else ["ls-files", "-z"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print("size-gate: git command failed; give paths", file=sys.stderr)
        sys.exit(2)
    names = [p for p in out.stdout.split("\0") if p]
    if not staged:
        return [(n, (base / n).read_bytes()) for n in names if (base / n).is_file()]
    entries = []
    for rel in names:
        show = subprocess.run(["git", "-C", str(base), "show", f":{rel}"], capture_output=True)
        if show.returncode == 0:
            entries.append((rel, show.stdout))
    return entries


def skipped(path):
    # Version control data and build trees hold generated files that are not the repository's.
    return any(part == ".git" or part.startswith("build") for part in path.parts)


def main():
    base = Path.cwd()
    args = sys.argv[1:]
    if args == ["--staged"]:
        entries = git_files(base, staged=True)
    elif args:
        entries = []
        for a in args:
            p = Path(a)
            if not p.exists():
                print(f"size-gate: no such path: {a}", file=sys.stderr)
                return 2
            files = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
            entries += [(str(q), q.read_bytes()) for q in files if not skipped(q)]
    else:
        entries = git_files(base, staged=False)
    findings, warnings = [], []
    for name, data in entries:
        if Path(name).suffix.lower() in SKIP_SUFFIXES:
            continue
        lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        limit = HOST_CEILING if name.endswith("_host.cu") else CEILING
        if lines > limit:
            findings.append(f"{name}: {lines} lines (limit {limit})")
        elif lines > WARNING:
            warnings.append(f"{name}: {lines} lines (warning at {WARNING})")
    for w in warnings:
        print(f"warning: {w}")
    for f in findings:
        print(f)
    if findings:
        print(f"size-gate: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"size-gate: clean ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
