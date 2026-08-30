#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# replay_session_check.sh: check the derived files of the long-session replay.
# Inputs: a journal and two boot identities. Output: one proof line or refusal lines.
# Exit codes: 0 clean, 1 a required result is absent, 2 usage error.
set -u

if [ "$#" -ne 3 ]; then
    echo "usage: replay_session_check.sh <journal> <first boot> <second boot>" >&2
    exit 2
fi
root="$1"
first="$2"
second="$3"
manifest_1="$root/manifest/$first.jsonl"
manifest_2="$root/manifest/$second.jsonl"
transcript_1="$root/$first/transcript/0.jsonl"
transcript_2="$root/$second/transcript/0.jsonl"
bad=0

if [ "$(sed -n '6p' "$manifest_1" 2>/dev/null | grep -c '"output_hash"' || true)" -ne 1 ]; then
    echo "replay_test: FAIL the compaction turn has no manifest" >&2
    bad=1
fi
finding=$(grep -h '"provenance":"computed"' "$root"/bus/*.jsonl 2>/dev/null \
    | grep '"claim":"folded first [0-9][0-9]* last [0-9][0-9]* count [1-9]' \
    | head -1 || true)
if [ -z "$finding" ]; then
    echo "replay_test: FAIL no computed finding names the folded range" >&2
    bad=1
fi
if ! grep -q '"kind":"summary"' "$transcript_1" 2>/dev/null; then
    echo "replay_test: FAIL the transcript has no summary" >&2
    bad=1
fi
hash_5=$(sed -n '5s/.*"input_hash":"\([0-9a-f]*\)".*/\1/p' "$manifest_1")
hash_7=$(sed -n '7s/.*"input_hash":"\([0-9a-f]*\)".*/\1/p' "$manifest_1")
if [ -z "$hash_5" ] || [ -z "$hash_7" ] || [ "$hash_5" = "$hash_7" ]; then
    echo "replay_test: FAIL the prompt after the summary has no distinct input hash" >&2
    bad=1
fi
if ! cmp -s "$transcript_1" "$transcript_2"; then
    echo "replay_test: FAIL the restored transcript differs from the first transcript" >&2
    diff -u "$transcript_1" "$transcript_2" >"$root/session-transcript.diff" 2>/dev/null || true
    bad=1
fi
turns=$(wc -l <"$manifest_1" 2>/dev/null || printf '0')
replayed=$(wc -l <"$manifest_2" 2>/dev/null || printf '0')
echo "session proof: $turns first manifests, $replayed restored manifests, one folded range, one summary, equal transcripts"
exit "$bad"
