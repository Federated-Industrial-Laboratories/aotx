#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create one instance through CTRL and check its complete live lifecycle.
# Inputs: Boot, client, model directory, module directory, and an empty run directory.
# Outputs: The full exchange in exchange.log. Exit codes: 0 pass, 1 check, 2 usage, 4 card refusal.
set -u

if [ "$#" -ne 5 ]; then
    echo "usage: live-check.sh <boot> <client> <models> <modules> <run-dir>" >&2
    exit 2
fi

boot=$(realpath "$1")
client=$(realpath "$2")
models=$(realpath "$3")
modules=$(realpath "$4")
run_dir=$5
socket_root=

finish()
{
    if [ -n "$socket_root" ] && [ -L "$socket_root" ]; then
        unlink "$socket_root"
    fi
}
trap finish EXIT INT TERM

if [ -e "$run_dir" ] && [ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "live check: the run directory is not empty" >&2
    exit 2
fi
mkdir -p "$run_dir"
run_dir=$(realpath "$run_dir")
socket_root=/tmp/aotx_ctrl_live_$$
ln -s "$run_dir" "$socket_root"
exchange=$run_dir/exchange.log
exec > >(tee "$exchange") 2>&1

echo "live check: module directory $modules"
for process in aotx_boot aotx_feed aotx_drain; do
    if pgrep -x "$process" >/dev/null; then
        echo "live check: $process is active"
        exit 4
    fi
done
card_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits)
card_status=$?
echo "live check: card query exit $card_status"
if [ "$card_status" -ne 0 ]; then
    exit 1
fi

large_holder=
aotx_holder=
while IFS=, read -r holder_pid holder_name holder_memory; do
    [ -n "$holder_pid" ] || continue
    holder_memory=${holder_memory//[[:space:]]/}
    holder_command=$(ps -p "$holder_pid" -o comm=)
    case "$holder_name $holder_command" in
        *aotx|*aotx_*) aotx_holder="$holder_pid,$holder_name,$holder_memory MiB" ;;
    esac
    if [ "$holder_memory" -gt 1024 ]; then
        large_holder="$holder_pid,$holder_name,$holder_memory MiB"
        break
    fi
done <<< "$card_apps"
if [ -n "$aotx_holder" ]; then
    echo "$aotx_holder"
    echo "live check: an AOTX card process is active"
    exit 4
fi
if [ -n "$large_holder" ]; then
    echo "$large_holder"
    echo "live check: a card holder is above one GiB"
    exit 4
fi

read -r free_memory < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
free_status=$?
if [ "$free_status" -ne 0 ] || [ -z "$free_memory" ]; then
    echo "live check: free memory query exit $free_status"
    exit 1
fi
free_memory=${free_memory//[[:space:]]/}
profile=$("$boot" --version | awk '{for (i = 1; i < NF; i++) if ($i == "profile") print $(i + 1)}')
case "$profile" in
    8g) profile_need=6398 ;;
    12g) profile_need=9574 ;;
    24g) profile_need=19062 ;;
    48g) profile_need=46230 ;;
    *) echo "live check: the build profile is not known"; exit 1 ;;
esac
echo "live check: profile $profile needs $profile_need MiB; $free_memory MiB is free"
if [ "$free_memory" -lt "$profile_need" ]; then
    echo "live check: the free memory does not cover the profile"
    exit 4
fi
if [ -n "$card_apps" ]; then
    echo "$card_apps"
fi
echo "live check: the card can start the profile"

export PATH=/usr/local/cuda-13.2/bin:$PATH
"$client" "$boot" "$models" "$socket_root"
client_status=$?
echo "live check: client exit $client_status"
if [ -f "$run_dir/journal/phase" ]; then
    echo -n "live check: final phase "
    cat "$run_dir/journal/phase"
fi
remaining=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)
remaining_status=$?
echo "live check: final card query exit $remaining_status"
if [ "$client_status" -ne 0 ] || [ "$remaining_status" -ne 0 ] ||
   ! grep -q '^closed ' "$run_dir/journal/phase"; then
    exit 1
fi
case "$remaining" in
    *aotx*) echo "live check: an AOTX card process remains"; exit 1 ;;
esac
exit 0
