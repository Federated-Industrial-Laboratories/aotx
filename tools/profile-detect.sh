#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# profile-detect.sh: propose the build profile and the architecture for the card in this
# machine.
# Input: none; the script reads the card with nvidia-smi.
# Output: one line, "profile <name> arch <n>", and the two configure options below it.
# Exit codes: 0 when a card was read, 2 when no card answered.
set -u

if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "profile-detect: nvidia-smi is not in the path" >&2
    exit 2
fi

read -r line < <(nvidia-smi --query-gpu=memory.total,compute_cap --format=csv,noheader) || true
if [ -z "${line:-}" ]; then
    echo "profile-detect: no card answered" >&2
    exit 2
fi

# The first field is the memory in mebibytes, the second is the compute capability.
total="$(echo "$line" | cut -d, -f1 | tr -dc '0-9')"
cap="$(echo "$line" | cut -d, -f2 | tr -dc '0-9.')"
if [ -z "$total" ] || [ -z "$cap" ]; then
    echo "profile-detect: the card gave no memory or no capability" >&2
    exit 2
fi

# The profile of a card is the largest one its memory holds. The figures follow the four
# profile headers in cuda/profile.
if   [ "$total" -ge 45000 ]; then profile="48g"
elif [ "$total" -ge 22000 ]; then profile="24g"
elif [ "$total" -ge 11000 ]; then profile="12g"
else                              profile="8g"
fi

# The architecture is the capability with the point taken out: 8.6 gives 86.
arch="$(echo "$cap" | tr -d '.')"

echo "card ${total} MiB capability ${cap}"
echo "profile ${profile} arch ${arch}"
echo "cmake -S . -B build -G Ninja -DAOTX_PROFILE=${profile} -DAOTX_ARCH=${arch}"
