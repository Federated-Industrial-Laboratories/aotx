#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# hang_tool.sh: run past the timeout of the manifest and write nothing.
# Inputs: one requests line, as JSON, on the standard input.
# Outputs: none; the check program must end this program.
# Exit codes: none that a caller reads; the program never ends by itself.
cat > /dev/null
sleep 300
