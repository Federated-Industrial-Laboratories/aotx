#!/usr/bin/env bash
# model_test.sh: Give CTest one skip status when a required model store is absent.
# Inputs: model directory, check program and its arguments. Output: the check output.

set -u
if [ "$#" -lt 2 ]; then
    echo "usage: model_test.sh <model directory> <check> [arguments]" >&2
    exit 2
fi
models="$1"
shift
if [ ! -r "$models/manifest.jsonl" ]; then
    echo "model check: skipped because $models/manifest.jsonl is not present"
    exit 77
fi
exec "$@"
