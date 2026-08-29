#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# echo_upper.sh: give the text argument of a tool call in capital letters.
# Inputs: one requests line, as JSON, on the standard input. The arg field holds key=value
#   pairs, and the unit separator byte comes before every pair. The drain writes that byte
#   as the escape of six characters, so this program takes the byte and the escape.
# Outputs: the value of the text argument in capital letters, on the standard output.
# Exit codes: 0 written, 1 when the line names no text argument.
set -u
line=$(cat)
value=$(printf '%s' "$line" | sed -n 's/.*"arg":"\([^"]*\)".*/\1/p')
text=$(printf '%s' "$value" | awk '
    {
        gsub(/\\u001f/, "\037")
        if (index($0, "\037") == 0) { print $0; exit }
        n = split($0, pair, "\037")
        for (i = 1; i <= n; i++) {
            if (substr(pair[i], 1, 5) == "text=") { print substr(pair[i], 6); exit }
        }
    }')
if [ -z "$text" ]; then
    echo "the line names no text argument" >&2
    exit 1
fi
printf '%s\n' "$text" | tr 'abcdefghijklmnopqrstuvwxyz' 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
