#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gate.sh: run every gate. With no arguments it examines staged content, for use as a
# pre-commit hook. With arguments it examines the given files or directories.
# Exit codes: 0 when every gate is clean, 1 when a gate has findings, 2 on an environment error.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -eq 0 ]; then set -- --staged; fi
status=0
# The register and size gates exclude third-party files below ctrl/vendor.
for gate in ste-lint.py size-gate.py seam-gate.py; do
    python3 "$here/$gate" "$@"
    rc=$?
    if [ "$rc" -eq 2 ]; then exit 2; fi
    if [ "$rc" -ne 0 ]; then status=1; fi
done
exit "$status"
