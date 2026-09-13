#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check device conditional classification and continued rejection of host and disk calls.
# Inputs: seam gate path. Output: case count. Exit: zero pass, nonzero failure.
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("seam_gate", Path(sys.argv[1]))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
cases = [
    ("cuda/policy/work.cu", "cudaGraphSetConditional(handle, value);", 0),
    ("cuda/policy/work.cu", "cudaGraphConditionalHandleCreate(&handle, graph, 0, 0);", 1),
    ("cuda/policy/work.cu", "cudaGraphSetConditionalExtra(handle, value);", 1),
    ("cuda/policy/work.cu", "cudaGraphSetConditional(h, 1); cudaMalloc(&p, 1);", 1),
    ("cuda/policy/work.cu", "cuGraphAddKernelNode(&node, graph, 0, 0, &p);", 1),
    ("cuda/policy/work.cu", "cudaDeviceSynchronize();", 1),
    ("disk/policy/file.c", "cudaGraphSetConditional(handle, value);", 1),
    ("cuda/policy/module_host.cu", "cudaGraphConditionalHandleCreate(&handle, graph, 0, 0);", 0),
]
for name, source, expected in cases:
    rules = gate.classify(name)
    actual = gate.scan(name, source.encode(), rules) if rules else []
    assert len(actual) == expected, (name, source, actual)
print(f"seam gate: {len(cases)} cases passed")
