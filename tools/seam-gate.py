#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# seam-gate.py: the seam gate.
#
# The gate keeps host work out of device files and CUDA out of disk-side files. A device
# file is a .cu or .cuh file whose name does not end in _host.cu. In a device file the gate
# refuses runtime calls, driver calls, managed memory, synchronization, device printf,
# device malloc, the C++ standard library and exceptions. In a .c or .h file under disk/ it
# refuses every CUDA symbol. Files under tests/ are host programs and are not examined.
#   seam-gate.py PATH [PATH...]   examine the given files or directories.
#   seam-gate.py --staged         examine staged content only.
#   seam-gate.py                  examine all git-tracked files under the current directory.
# Exit codes: 0 clean, 1 findings, 2 usage or environment error.

import ast
import re
import subprocess
import sys
from pathlib import Path

# This conditional control call is device-only in the CUDA runtime interface.
DEVICE_FORBIDDEN = [
    re.compile(r"\bcuda(?!GraphSetConditional\b)[A-Z]\w*\s*\("),
    re.compile(r"\bcu[A-Z]\w*\s*\("),
    re.compile(r"__managed__"),
    re.compile(r"\bprintf\s*\("),
    re.compile(r"\bmalloc\s*\("),
    re.compile(r"\bfree\s*\("),
    re.compile(r"\bnew\s+[A-Za-z_]"),
    re.compile(r"\bdelete\b"),
    re.compile(r"\bfopen\s*\("),
    re.compile(r"\bstd::"),
    re.compile(r"#include\s*<(iostream|fstream|sstream|vector|string|map|set|thread|mutex|memory|algorithm)>"),
    re.compile(r"\bthrow\b|\btry\b|\bcatch\b"),
    re.compile(r"\bint\s+main\s*\("),
]
DISK_FORBIDDEN = [
    re.compile(r"\bcuda\w*"),
    re.compile(r"\bcu[A-Z]\w*\s*\("),
    re.compile(r"__global__|__device__|__host__"),
    re.compile(r"<<<"),
]
STRING_OR_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)
GATEWAY_IMPORTS = {'argparse', 'asyncio', 'base64', 'binascii', 'codecs', 'contextlib',
    'dataclasses', 'hashlib', 'hmac', 'ipaddress', 'json', 'logging', 'math', 'os', 're',
    'signal', 'socket', 'ssl', 'stat', 'struct', 'tempfile', 'time', 'uuid', 'aiohttp', 'yarl'}
GATEWAY_MODULES = {'controls', 'capabilities', 'config', 'errors', 'fetch', 'json_wire', 'limits',
    'media', 'output', 'policy', 'requests', 'server', 'wire', 'shared', 'shared_wire', 'shared_output'}


def gateway_scan(name, data):
    findings = []
    try: tree = ast.parse(data.decode('utf-8'))
    except (UnicodeError, SyntaxError): return [f'{name}: invalid Python source']
    for node in ast.walk(tree):
        reason = None
        if isinstance(node, ast.Import):
            if any(a.name.split('.')[0] not in GATEWAY_IMPORTS for a in node.names): reason = 'unapproved gateway dependency'
        elif isinstance(node, ast.ImportFrom):
            allowed = GATEWAY_MODULES if node.level == 1 else GATEWAY_IMPORTS
            if node.level > 1 or not node.module or node.module.split('.')[0] not in allowed:
                reason = 'unapproved gateway dependency'
        elif isinstance(node, ast.Call):
            target = node.func
            word = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ''
            if word in {'eval', 'exec', 'compile', '__import__', 'system', 'popen', 'fork', 'forkpty', 'posix_spawn', 'posix_spawnp'} or word.startswith(('execv', 'execl', 'spawnv', 'spawnl')):
                reason = 'dynamic code or process execution in the gateway'
        if reason: findings.append(f'{name}:{node.lineno}: {reason}')
    return findings


def git_files(base, staged):
    cmd = ["git", "-C", str(base)]
    cmd += ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged else ["ls-files", "-z"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print("seam-gate: git command failed; give paths", file=sys.stderr)
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


def classify(name):
    parts = Path(name).parts
    if "tests" in parts:
        return None
    suffix = Path(name).suffix.lower()
    if suffix in {".cu", ".cuh"} and not name.endswith("_host.cu"):
        return DEVICE_FORBIDDEN
    if suffix in {".c", ".h"} and "disk" in parts:
        return DISK_FORBIDDEN
    return None


def scan(name, data, rules):
    findings = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{name}: not UTF-8"]
    # Strings and comments are blanked so that a word in a comment is not a call.
    blanked = STRING_OR_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    for lineno, line in enumerate(blanked.splitlines(), 1):
        for rule in rules:
            m = rule.search(line)
            if m:
                findings.append(f"{name}:{lineno}: '{m.group(0).strip()}' is on the wrong side of the seam")
    return findings


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
                print(f"seam-gate: no such path: {a}", file=sys.stderr)
                return 2
            files = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
            entries += [(str(q), q.read_bytes()) for q in files if not skipped(q)]
    else:
        entries = git_files(base, staged=False)
    findings, examined, headers = [], 0, []
    for name, data in entries:
        if 'gateway' in Path(name).parts and Path(name).suffix == '.py':
            examined += 1
            findings += gateway_scan(name, data)
            continue
        rules = classify(name)
        if rules is None:
            if "tools" in Path(name).parts and Path(name).suffix == ".h": headers.append(name)
            continue
        examined += 1
        findings += scan(name, data, rules)
    if headers:
        print("seam-gate: headers outside this check: " + ", ".join(headers))
    for f in findings:
        print(f)
    if findings:
        print(f"seam-gate: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"seam-gate: clean ({examined} files examined)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
