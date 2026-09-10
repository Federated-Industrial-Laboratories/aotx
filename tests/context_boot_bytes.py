#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Build independent contextual memory and correction bytes for the live boot test.
# Owns: Explicit private subjects, task cues, source events and appraisals.
# Threading: Disk-side fixture construction; no model computation.
# Lifetime: One integration test and its portable input files.

# Inputs: fixture module, batch and store cut. Output: wire bytes. Exit: caller exceptions.
import struct

from capacity_boot_test import batch, memory, color


def identity(i, part):
    return 500000 + i * 64 + part


def contextual(f, i, text, required=True):
    data = bytearray(64) + text.encode()
    data[:8] = b"AOTXMEM2"; f.put(data, 8, 2, 4); f.put(data, 12, len(text.encode()), 4)
    data[16:32] = f.identity(40000 + i); f.put(data, 32, int(required), 4)
    return data


def source(f, row, i, part):
    row[96:112] = f.identity(identity(i, part)); f.put(row, 112, 1)


def corpus(f, count, text):
    values = []
    for i in range(count):
        for part, kind in enumerate((1, 2, 5, 9, 2, 3)):
            if text and part >= 3:
                continue
            r = f.object_row(kind, identity(i, part), 10000 + i, len(values) + 1)
            r[120:136] = f.identity(30000 + i)
            if part == 0:
                p = memory(f, f"Member {i} supplied the check requirement.")
            elif part in (1, 2, 4):
                source(f, r, i, 0)
                if part == 1:
                    label = f"Member {i}: The required check color is {color(i)}."
                elif part == 2:
                    r[120:136] = bytes(16)
                    label = "For this task, use only the current member requirement. If no requirement exists, the check color is unknown."
                else:
                    label = f"Member {i}: A prior check contained both a useful result and an error."
                    r[208:224] = f.identity(identity(i, 3)); f.put(r, 224, 1); f.put(r, 180, 4, 4)
                p = contextual(f, i, label, part != 4)
            elif part == 3:
                source(f, r, i, 0); f.put(r, 180, 4, 4)
                p = bytearray(128) + struct.pack("<3f", 1.0, i / 128.0, 0.0)
                p[:8] = b"AOTXVEC2"
                for offset, value in ((8, 2), (12, 3), (16, 4), (20, 1)):
                    f.put(p, offset, value, 4)
                p[24:56], p[56:88], p[88:104] = f.MODEL, f.PROCESSOR, f.identity(identity(i, 0))
                f.put(p, 104, 1)
            else:
                source(f, r, i, 4); f.put(r, 180, 4, 4)
                p = bytearray(32)
                for offset, value in ((0, 1), (4, 700000 + i), (8, 900000 - i), (12, 600000),
                                      (16, 3), (20, 0xFFFFFFFF), (24, 1)):
                    f.put(p, offset, value, 4)
            values.append((r, p))
    return values


def correction(f, count, cut):
    values = []
    for i in range(count):
        r = f.object_row(1, identity(i, 6), 10000 + i, cut + len(values) + 1)
        r[120:136] = f.identity(30000 + i)
        values.append((r, memory(f, f"Member {i} corrected the check requirement.")))
        r = f.object_row(2, identity(i, 7), 10000 + i, cut + len(values) + 1)
        r[120:136] = f.identity(30000 + i); source(f, r, i, 6)
        r[136:152] = f.identity(identity(i, 1)); f.put(r, 152, 1)
        values.append((r, contextual(f, i, f"Member {i}: The required check color is {color(i, True)}.")))
    return f.image(values, cut + 1, 20, True)


def request(f, count, cut, turn, text):
    data = batch(f, b"AOTXTXT1" if text else b"AOTXLIV1", 8256, cut, count)
    for i in range(count):
        base = 64 + i * 8256; f.put(data, base, i, 4)
        data[base + 16:base + 32] = f.identity(8000 + i); f.put(data, base + 32, turn)
        q = bytearray(8192)
        q[:16], q[16:32], q[48:64] = f.identity(100000 + turn * 64 + i), f.identity(10000 + i), f.identity(200000 + turn * 64 + i)
        label = f"Member {i if turn != 2 else 1000 + i}: State the required check color from current memory. Reply with one word."
        if not text:
            q[64:96], q[96:128], q[160:172] = f.MODEL, f.PROCESSOR, struct.pack("<3f", 1.0, i / 128.0, 0.0)
            f.put(q, 128, 3, 4)
        for offset, value in ((132, 1 if turn == 2 else 2 if text else 4), (136, 2048), (148, len(label))):
            f.put(q, offset, value, 4)
        q[4640:4640 + len(label)] = label.encode()
        c = bytearray(1504); c[:8] = b"AOTXCTX1"
        for offset, value in ((8, 1), (12, 1 if text else 3), (32, 1), (44, 1)):
            f.put(c, offset, value, 4)
        c[16:32], c[48:64] = f.identity(40000 + i), f.identity((70000 if turn == 2 else 30000) + i)
        if not text:
            f.put(c, 36, 600000, 4); f.put(c, 40, 800000, 4)
        q[6688:] = c; data[base + 64:base + 8256] = q
    return data


def selected(f, i, turn, text):
    parts = [2] if turn == 2 else [1, 2] if turn == 1 else [2, 7]
    if not text and turn != 2:
        parts += [4, 5]
    return [f.identity(identity(i, p)) for p in parts]
