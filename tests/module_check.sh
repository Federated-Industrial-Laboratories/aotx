#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# module_check.sh: run the module check program over one module and compare the verdict.
# Inputs: the check program, the module directory, the verdict (pass or fail), the rows.
# Outputs: the lines of the check program on the standard output.
# Exit codes: 0 when the verdict is the one given, 1 when it is not, 2 usage error.
set -u
if [ "$#" -lt 3 ]; then
    echo "usage: module_check.sh <program> <module-dir> <pass|fail> [rows]" >&2
    exit 2
fi
program="$1"
dir="$2"
want="$3"
rows="${4:-}"
if [ -n "$rows" ]; then
    "$program" "$dir" "$rows"
else
    "$program" "$dir"
fi
status=$?
if [ "$want" = "pass" ] && [ "$status" -eq 0 ]; then
    echo "module_check: $(basename "$dir") passed, as the check asks"
    exit 0
fi
if [ "$want" = "fail" ] && [ "$status" -ne 0 ]; then
    echo "module_check: $(basename "$dir") was refused, as the check asks"
    exit 0
fi
echo "module_check: $(basename "$dir") gave the status $status and the check wants $want" >&2
exit 1
