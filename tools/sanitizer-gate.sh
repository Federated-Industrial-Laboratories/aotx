#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# sanitizer-gate.sh: the sanitizer gate.
#
# The gate runs compute-sanitizer over the checks that drive the tick graph. It refuses a
# run in which the sanitizer reports a memory error or a data hazard.
#
#   sanitizer-gate.sh TOOL BUILD [MODELS] [FIXTURES]
#     TOOL      memcheck or racecheck
#     BUILD     the directory that holds the check programs
#     MODELS    the directory of the model files; the default is models
#     FIXTURES  the directory of the tokenizer fixtures
#
# Input: the environment variable AOTX_SANITIZER_BIN names compute-sanitizer; without it
# the gate takes the one on the path. The environment variable AOTX_SANITIZER_SKIP holds
# the names of programs to leave out, with spaces between them.
# Output: one line for each program with its figures, then a summary line.
# Exit codes: 0 clean, 1 a finding or a check that failed, 2 usage or environment error.

# The gate sets AOTX_SANITIZER to the tool name for each run. The checks read it and take
# fewer ticks, because the sanitizer makes every kernel far slower than the tick budget
# allows. A check that must take a smaller batch under one tool reads the name.

# Each arm takes its own programs. The memcheck arm reads every access to memory. It takes
# the checks that drive the tick graph: the seam check, the decode check and the agent
# check.

# The racecheck arm reads every access to shared memory. The matrix kernels of the language
# model make the decode check run past 20 minutes under it. That arm therefore takes the
# checks that use shared memory by hand at a bounded size. Those are the seam check, the
# interface check and the matrix check.

# No program of this gate opens a window.

set -u

tool="${1:-}"
build="${2:-}"
models="${3:-models}"
fixtures="${4:-tests/fixtures/tokenizer}"

if [ -z "$tool" ] || [ -z "$build" ]; then
    echo "sanitizer-gate: give a tool and a build directory" >&2
    exit 2
fi
if [ "$tool" != "memcheck" ] && [ "$tool" != "racecheck" ]; then
    echo "sanitizer-gate: the tool is memcheck or racecheck" >&2
    exit 2
fi

sanitizer="${AOTX_SANITIZER_BIN:-}"
if [ -z "$sanitizer" ]; then
    sanitizer=$(command -v compute-sanitizer || true)
fi
if [ -z "$sanitizer" ] || [ ! -x "$sanitizer" ]; then
    echo "sanitizer-gate: compute-sanitizer is not there" >&2
    exit 2
fi

# The programs of each arm. The seam check takes a short run, because the sanitizer holds
# every tick.
if [ "$tool" = "memcheck" ]; then
    programs="aotx_seam_device_test aotx_decode_device_test aotx_agent_device_test"
else
    programs="aotx_seam_device_test aotx_ui_test aotx_matrix_device_test"
fi
skip="${AOTX_SANITIZER_SKIP:-}"

ran=0
skipped=0
findings=0

for name in $programs; do
    program="$build/$name"
    if [ ! -x "$program" ]; then
        echo "sanitizer-gate: $name is not built; skipped"
        skipped=$((skipped + 1))
        continue
    fi
    left=0
    for out in $skip; do
        [ "$out" = "$name" ] && left=1
    done
    if [ "$left" -eq 1 ]; then
        echo "sanitizer-gate: $name is in AOTX_SANITIZER_SKIP; skipped"
        skipped=$((skipped + 1))
        continue
    fi
    case "$name" in
        aotx_seam_device_test) set -- --workload 100 --seconds 1 ;;
        aotx_decode_device_test) set -- "$models" "$fixtures" ;;
        aotx_matrix_device_test) set -- "$models" ;;
        *) set -- ;;
    esac
    log="$build/sanitizer-$tool-$name.log"
    AOTX_SANITIZER="$tool" "$sanitizer" --tool "$tool" --error-exitcode 1 --print-limit 20 \
        "$program" "$@" > "$log" 2>&1
    status=$?
    ran=$((ran + 1))
    summary=$(grep -E "ERROR SUMMARY|RACECHECK SUMMARY" "$log" | tail -1)
    if [ -z "$summary" ]; then
        summary="no summary line"
    fi
    echo "sanitizer-gate: $tool $name exit $status, $summary"
    if [ "$status" -ne 0 ]; then
        findings=$((findings + 1))
        grep -E "^========= (Invalid|Race|Program hit|Error|Barrier)" "$log" | head -20
        echo "sanitizer-gate: the whole output is in $log"
    fi
done

echo "sanitizer-gate: $tool over $ran program(s), $skipped skipped, $findings finding(s)"
if [ "$ran" -eq 0 ]; then
    echo "sanitizer-gate: no program ran" >&2
    exit 2
fi
[ "$findings" -eq 0 ] || exit 1
exit 0
