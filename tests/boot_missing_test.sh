#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Check that a named settings file is mandatory.
# Input: The boot program path.
# Output: One result line; status 0 passes, status 1 fails, and status 2 is bad use.

if [ "$#" -ne 1 ]; then
    echo "boot_missing_test: give the boot program" >&2
    exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
missing="$work/not-there.settings"
report=$($1 --settings "$missing" 2>&1)
status=$?

if [ "$status" -ne 2 ]; then
    echo "boot_missing_test: FAIL the refusal status is $status" >&2
    exit 1
fi
case "$report" in
    *"settings: the file is not there: $missing"*) ;;
    *) echo "boot_missing_test: FAIL the refusal does not name the file" >&2; exit 1 ;;
esac
case "$report" in
    *"clock module:"*|*"boot: id"*)
        echo "boot_missing_test: FAIL the system started" >&2
        exit 1
        ;;
esac

echo "boot_missing_test: the named file was refused before the start"
