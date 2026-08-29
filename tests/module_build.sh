#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# module_build.sh: run the module build script over one module and compare the verdict.
# Inputs: the source directory, the module directory, the verdict (pass or fail), the arch.
# Outputs: the lines of the build script on the standard output.
# Exit codes: 0 when the verdict is the one given, 1 when it is not, 2 usage error.
set -u
if [ "$#" -lt 4 ]; then
    echo "usage: module_build.sh <source-dir> <module-dir> <pass|fail> <arch>" >&2
    exit 2
fi
source_dir="$1"
dir="$2"
want="$3"
arch="$4"
bash "$source_dir/tools/module-build.sh" "$dir" --arch "$arch"
status=$?
if [ "$want" = "pass" ] && [ "$status" -eq 0 ]; then
    echo "module_build: $(basename "$dir") built, as the check asks"
    exit 0
fi
if [ "$want" = "fail" ] && [ "$status" -ne 0 ]; then
    echo "module_build: $(basename "$dir") was refused, as the check asks"
    exit 0
fi
echo "module_build: $(basename "$dir") gave the status $status and the check wants $want" >&2
exit 1
