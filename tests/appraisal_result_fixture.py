#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Read complete appraisal result frames without model generation.
# Owns: Independent version, row, output and call-boundary checks.
# Threading: One test caller reads a complete recorded source batch.
# Lifetime: One immutable result frame.

# Inputs: Complete bytes. Output: Header, rows and memory tail. Exit: ValueError on invalid input.
import json
import struct


def require(value, message):
    if not value:
        raise ValueError(message)


def u32(data, at):
    return struct.unpack_from("<I", data, at)[0]


def u64(data, at):
    return struct.unpack_from("<Q", data, at)[0]


def decode_evidence(raw, source):
    require(0 < len(raw) <= 4096, "Invalid evidence size.")
    quotes = json.loads(raw.decode("utf-8"))
    require(isinstance(quotes, list) and len(quotes) <= 8, "Invalid evidence array.")
    require(all(isinstance(quote, str) and quote and source.count(quote.encode("utf-8")) == 1
                for quote in quotes), "Invalid source quote.")
    require(len(set(quotes)) == len(quotes), "Repeated evidence quote.")
    return quotes


def decode_result(data):
    require(len(data) >= 64 and data[:8] == b"AOTXAPS1", "Invalid result header.")
    version, count, status = u32(data, 8), u32(data, 12), u32(data, 32)
    require(version in (1, 2), "Unsupported result version.")
    stride = 4160 if version == 1 else 8320
    tail_size = u64(data, 24)
    require(count <= 64 and status <= 12 and not any(data[36:64]), "Invalid result fields.")
    require(len(data) == 64 + count * stride + tail_size, "Invalid result size.")
    require(count or status and not tail_size, "An empty result requires a refusal.")
    rows, seen = [], set()
    for index in range(count):
        row = data[64 + index * stride:64 + (index + 1) * stride]
        identity, revision, model = bytes(row[:16]), u64(row, 16), bytes(row[24:56])
        length = u32(row, 56)
        require(any(identity) and revision and identity not in seen, "Invalid or repeated source queue.")
        seen.add(identity)
        require(u32(row, 60) == status and length <= 4096, "Invalid result row.")
        require(not any(row[64 + length:4160]), "Invalid response padding.")
        require(status or length and any(model), "A complete result requires output and a model.")
        first, first_model, phase, second = b"", bytes(32), 0, 0
        if version == 2:
            first_size, second, phase = u32(row, 4160), u32(row, 4164), u32(row, 4168)
            first_model = bytes(row[4192:4224])
            require(first_size <= 4096 and second <= 1 and phase <= 2, "Invalid outcome row.")
            require(not any(row[4172:4192]) and not any(row[4224 + first_size:]), "Invalid outcome padding.")
            first = bytes(row[4224:4224 + first_size])
            require(not any(first_model) or first_model == model, "The two calls use different models.")
            require(not first_size or any(first_model), "Outcome output requires its model.")
            if phase == 0:
                require(not first_size and not length and not second and not any(first_model), "Output before a model call.")
            elif phase == 1:
                require(not length and not second, "Score output before the second call.")
            else:
                require(first_size and any(first_model) and (second or not length), "Invalid second-call boundary.")
            require(status or phase == 2 and second == 1, "A complete result requires both calls.")
        rows.append(dict(queue=identity, version=revision, model=model, reply=bytes(row[64:64 + length]),
                         first=first, first_model=first_model, phase=phase, second_call=second, status=status))
    return dict(version=version, sequence=u64(data, 16), status=status, rows=rows,
                tail=bytes(data[64 + count * stride:]))
