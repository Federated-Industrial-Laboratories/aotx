#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# spill-gate.py: the spill gate.
#
# The gate reads the resource usage that the device linker recorded. It refuses a hot
# kernel that keeps local memory or a stack frame over its allowance. It runs cuobjdump
# over each given file, and over each file of a given directory that holds device code.

#   spill-gate.py PATH [PATH...]     examine the given files and directories.
#   spill-gate.py --list             print the kernel list and the allowances.
#   spill-gate.py --cuobjdump PATH   the cuobjdump to run; the default comes from PATH.

# Output: one line for each kernel of the list that was found, with the file that holds
# the largest figures. Then one line for each finding, then a summary line.
# Exit codes: 0 clean, 1 findings, 2 usage or environment error.

# STACK is the stack frame the kernel keeps. It holds the spill slots and the outgoing
# argument space of a call that the compiler did not inline. LOCAL is local memory, which
# a spilled array gives.

# A call across translation units is not inlined, so the caller keeps a frame for it even
# with no spill. The allowance of such a kernel is the frame that its calls need, and it
# is stated with the reason.

# Limit of this gate, stated: a spill that stays inside a frame that is already allowed
# does not raise the figure. The kernels whose allowance is zero have no such cover.

import re
import shutil
import subprocess
import sys
from pathlib import Path

# The hot kernels and the bytes of stack frame each may keep. A kernel with an allowance
# above zero calls a device function in another translation unit, and the figure is the
# frame that call needs. Local memory is refused in every one of them.
ALLOWANCE = {
    "aotx_checkpoint_fill":    (0,   "parallel snapshot byte copies"),
    "aotx_checkpoint_copy":    (0,   "bounded mapped transport copy"),
    "aotx_checkpoint_publish": (0,   "completed slot publication"),
    "aotx_checkpoint_step":    (88,  "frames of the checkpoint encoder and byte reader calls"),
    "aotx_model_gemm":         (0,   "tensor core product; every operand is in registers"),
    "aotx_model_gemv":         (0,   "memory bound product; every operand is in registers"),
    "aotx_model_dequant":      (0,   "block reader; no call and no array"),
    "aotx_model_attend":       (0,   "attention over the page table"),
    "aotx_model_qkv":          (32,  "frame of a call across translation units"),
    "aotx_model_norm":         (0,   "root mean square scale"),
    "aotx_model_swiglu":       (0,   "gate and up product"),
    "aotx_model_gather":       (0,   "token embedding gather"),
    "aotx_model_product":      (0,   "tensor core node of a matrix product of the decode"),
    "aotx_model_line":         (0,   "memory bound node of a matrix product of the decode"),
    "aotx_decode_plan":        (0,   "the batch of one tick"),
    "aotx_decode_commit":      (88,  "frames of calls that detokenize sampled tokens"),
    "aotx_model_open_rows":    (0,   "row open"),
    "aotx_model_pick":         (0,   "sampling"),
    "aotx_embed_pool":         (0,   "pooling and normalization"),
    "aotx_rerank_score":       (0,   "two logit softmax"),
    "aotx_seam_flush":         (0,   "record flush to the host ring"),
    "aotx_seam_apply_inbound": (520, "frames of the calls across translation units"),
    "aotx_seam_bulk_flush":    (0,   "bulk flush to the host ring"),
    "aotx_text_merge":         (0,   "byte pair merges, one warp for each chunk"),
    "aotx_text_pretok":        (24,  "state machine over the seven alternatives"),
    "aotx_agent_step":         (296, "frames of calls and the bounded prompt builder"),
    "aotx_tool_step":          (288, "frames of the calls across translation units"),
    "aotx_tool_parse":         (0,   "tool call parser; it keeps no array of its own"),
    "aotx_tool_plan":          (0,   "the batch of the embedding pass of one tick"),
    "aotx_tool_fill":          (144, "the frame of the call that fills the module rows"),
    "aotx_embed_search":       (0,   "cosine search over the note store"),
}

SKIP_SUFFIXES = {".txt", ".cmake", ".json", ".ninja", ".log", ".py", ".sh", ".md", ".gguf",
                 ".bin", ".c", ".cu", ".h", ".cuh", ".jsonl"}
ELF_MAGIC = b"\x7fELF"
AR_MAGIC = b"!<arch>\n"
FUNCTION = re.compile(r"^\s*Function\s+(\S+?):\s*$")
FIGURE = re.compile(r"(STACK|LOCAL):(\d+)")
MANGLED = re.compile(r"^_Z(\d+)([A-Za-z_].*)$")


def kernel_name(symbol):
    # A kernel symbol is _Z, the length of the name, then the name and the argument types.
    # The length gives the name exactly, so no other symbol can match by a prefix.
    found = MANGLED.match(symbol)
    if found:
        return found.group(2)[:int(found.group(1))]
    return symbol


def readable(path):
    if path.suffix in SKIP_SUFFIXES or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    return head.startswith(ELF_MAGIC) or head.startswith(AR_MAGIC)


def targets(paths):
    files = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files += sorted(one for one in path.iterdir() if readable(one))
        elif readable(path):
            files.append(path)
        else:
            print(f"spill-gate: {name} holds no device code", file=sys.stderr)
            sys.exit(2)
    return files


def usage(tool, path):
    # One reading of one file. The result is the largest STACK and LOCAL of each kernel.
    # One file can hold the same kernel more than one time.
    out = subprocess.run([tool, "--dump-resource-usage", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    found = {}
    name = None
    for line in out.stdout.splitlines():
        head = FUNCTION.match(line)
        if head:
            name = kernel_name(head.group(1))
            continue
        if name is None or "STACK:" not in line:
            continue
        if name in ALLOWANCE:
            figures = {key: int(value) for key, value in FIGURE.findall(line)}
            stack = figures.get("STACK", 0)
            local = figures.get("LOCAL", 0)
            held = found.get(name, (0, 0))
            found[name] = (max(held[0], stack), max(held[1], local))
        name = None
    return found


def main(argv):
    tool = "cuobjdump"
    paths = []
    at = 0
    while at < len(argv):
        if argv[at] == "--list":
            for name in sorted(ALLOWANCE):
                allowed, why = ALLOWANCE[name]
                print(f"{name}: {allowed} bytes of stack frame, no local memory ({why})")
            return 0
        if argv[at] == "--cuobjdump":
            at += 1
            if at >= len(argv):
                print("spill-gate: --cuobjdump needs a path", file=sys.stderr)
                return 2
            tool = argv[at]
        else:
            paths.append(argv[at])
        at += 1
    if not paths:
        print("spill-gate: give files or directories", file=sys.stderr)
        return 2
    if shutil.which(tool) is None and not Path(tool).is_file():
        print(f"spill-gate: {tool} not found", file=sys.stderr)
        return 2

    files = targets(paths)
    if not files:
        print("spill-gate: no file holds device code", file=sys.stderr)
        return 2

    worst = {}
    for path in files:
        for name, (stack, local) in usage(tool, path).items():
            held = worst.get(name, (0, 0, ""))
            if stack > held[0] or local > held[1] or held[2] == "":
                keep = (max(stack, held[0]), max(local, held[1]), path.name)
                worst[name] = keep
    if not worst:
        print("spill-gate: none of the kernels of the list are in these files",
              file=sys.stderr)
        return 2

    findings = 0
    for name in sorted(worst):
        stack, local, where = worst[name]
        allowed = ALLOWANCE[name][0]
        mark = "ok" if (stack <= allowed and local == 0) else "REFUSED"
        print(f"spill-gate: {name}: STACK {stack} of {allowed}, LOCAL {local} of 0, "
              f"in {where} [{mark}]")
        if stack > allowed:
            print(f"spill-gate: {name} keeps {stack} bytes of stack frame over the "
                  f"allowance of {allowed} ({where})", file=sys.stderr)
            findings += 1
        if local != 0:
            print(f"spill-gate: {name} keeps {local} bytes of local memory ({where})",
                  file=sys.stderr)
            findings += 1
    missing = sorted(set(ALLOWANCE) - set(worst))
    if missing:
        print(f"spill-gate: not built, and not examined: {', '.join(missing)}")
    print(f"spill-gate: {len(worst)} kernels of {len(ALLOWANCE)} examined in "
          f"{len(files)} files, {findings} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
