#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Drive all six first-run pages against one live boot from a clean home.
# Inputs: Boot, walker, model directory, and an empty run directory.
# Outputs: Page gates and the first reply. Exit codes: 0 pass, 1 check, 2 usage, 4 card refusal.
set -u

if [ "$#" -ne 4 ]; then
    echo "usage: wizard-walk.sh <boot> <walker> <models> <run-dir>" >&2
    exit 2
fi

boot=$(realpath "$1")
walker=$(realpath "$2")
models=$(realpath "$3")
run_dir=$4
socket_root=

finish()
{
    if [ -n "$socket_root" ] && [ -L "$socket_root" ]; then
        unlink "$socket_root"
    fi
}
trap finish EXIT INT TERM

if [ -e "$run_dir" ] && [ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "wizard walk: the run directory is not empty" >&2
    exit 2
fi
mkdir -p "$run_dir"
run_dir=$(realpath "$run_dir")
socket_root=/tmp/aotx_ctrl_wizard_$$
ln -s "$run_dir" "$socket_root"
mkdir -p "$run_dir/home"

for process in aotx_boot aotx_feed aotx_drain; do
    if pgrep -x "$process" >/dev/null; then
        echo "wizard walk: $process is active"
        exit 4
    fi
done
card_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits)
card_status=$?
echo "wizard walk: card query exit $card_status"
if [ "$card_status" -ne 0 ]; then
    exit 1
fi
while IFS=, read -r holder_pid holder_name holder_memory; do
    [ -n "$holder_pid" ] || continue
    holder_memory=${holder_memory//[[:space:]]/}
    holder_command=$(ps -p "$holder_pid" -o comm=)
    case "$holder_name $holder_command" in
        *aotx|*aotx_*) echo "wizard walk: an AOTX card process is active"; exit 4 ;;
    esac
    if [ "$holder_memory" -gt 1024 ]; then
        echo "wizard walk: a card holder is above one GiB"
        exit 4
    fi
done <<< "$card_apps"

read -r free_memory < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
free_memory=${free_memory//[[:space:]]/}
profile=$("$boot" --version | awk '{for (i = 1; i < NF; i++) if ($i == "profile") print $(i + 1)}')
case "$profile" in
    8g) profile_need=6398 ;;
    12g) profile_need=9574 ;;
    24g) profile_need=19062 ;;
    48g) profile_need=46230 ;;
    *) echo "wizard walk: the build profile is not known"; exit 1 ;;
esac
echo "wizard walk: profile $profile needs $profile_need MiB; $free_memory MiB is free"
if [ "$free_memory" -lt "$profile_need" ]; then
    echo "wizard walk: the free memory does not cover the profile"
    exit 4
fi
echo "wizard walk: the card can start the profile"

export PATH=/usr/local/cuda-13.2/bin:$PATH
HOME="$run_dir/home" "$walker" "$boot" "$models" "$socket_root"
walk_status=$?
echo "wizard walk: client exit $walk_status"
remaining=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)
remaining_status=$?
echo "wizard walk: final card query exit $remaining_status"
if [ "$walk_status" -ne 0 ] || [ "$remaining_status" -ne 0 ]; then
    exit 1
fi
case "$remaining" in
    *aotx*) echo "wizard walk: an AOTX card process remains"; exit 1 ;;
esac
exit 0
