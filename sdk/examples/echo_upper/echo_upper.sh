#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# echo_upper.sh: give the text argument of a tool call in capital letters.
# Inputs: one requests line, as JSON, on the standard input.
# Outputs: the value of the text argument in capital letters, on the standard output.
# Exit codes: 0 written, 1 when the line names no text argument.
set -u
line=$(cat)
value=$(printf '%s' "$line" | sed -n 's/.*"arg":"\([^"]*\)".*/\1/p')
text=${value#text=}
if [ -z "$text" ] || [ "$text" = "$value" ]; then
    echo "the line names no text argument" >&2
    exit 1
fi
printf '%s\n' "$text" | tr 'abcdefghijklmnopqrstuvwxyz' 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
