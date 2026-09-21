#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Write pinned sentence properties and official boundary test vectors.
# Inputs: Unicode 17 property and test files. Outputs: device table and test header.
# Exit codes: 0 on success; 1 on a file or identity error.
import argparse
import hashlib
from pathlib import Path

CLASSES = ('Other', 'CR', 'LF', 'Extend', 'Sep', 'Format', 'Sp', 'Lower',
           'Upper', 'OLetter', 'Numeric', 'ATerm', 'STerm', 'Close', 'SContinue')
PROPERTY = '871c0c985ad95125e25b302414065a10839d068970bceb383ecec138f22a0a18'
TEST = '12cb47d028ded0c1cb8a28558f95479cbcd24559c46977015c82f3b50a1cc6e4'
BANNER = ('/* Purpose: Supply pinned Unicode 17 sentence boundary data.\n'
          ' * Owns: Generated data from the official property and test files.\n'
          ' * Launch shape: Batched source rows; the data has no mutable state.\n'
          ' * Lifetime: The whole run. */\n')


def read(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('input file identity differs')
    return data.decode().splitlines()


def main():
    parser = argparse.ArgumentParser(description='Write pinned sentence boundary data.')
    parser.add_argument('input', type=Path)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = []
    for line in read(args.input / 'SentenceBreakProperty.txt', PROPERTY):
        body = line.split('#')[0].strip()
        if not body:
            continue
        points, kind = map(str.strip, body.split(';'))
        ends = [int(x, 16) for x in points.split('..')]
        rows.append((ends[0], ends[-1], CLASSES.index(kind)))
    merged = []
    for first, last, kind in sorted(rows):
        if merged and merged[-1][1] + 1 == first and merged[-1][2] == kind:
            merged[-1] = (merged[-1][0], last, kind)
        else:
            merged.append((first, last, kind))
    out = [BANNER, '#include "cognitive/source_spans.cuh"\n',
           '__device__ const uint32_t aotx_source_properties[][2] = {\n']
    for at in range(0, len(merged), 8):
        out.append('    ' + ' '.join('{0x%x,0x%x},' % (a, (b << 4) | c)
                                   for a, b, c in merged[at:at + 8]) + '\n')
    out += ['};\n', f'__device__ const uint32_t aotx_source_property_count = {len(merged)};\n']
    (args.root / 'cuda/cognitive/source_table.cu').write_text(''.join(out))
    cases = []
    for line in read(args.input / 'SentenceBreakTest.txt', TEST):
        body = line.split('#')[0].strip()
        if not body:
            continue
        data, bounds = bytearray(), []
        for word in body.split():
            if word == '\u00f7':
                bounds.append(len(data))
            elif word != '\u00d7':
                data.extend(chr(int(word, 16)).encode())
        cases.append((bytes(data), bounds))
    if len(cases) != 512:
        raise ValueError('test count differs')
    out = [BANNER, '#ifndef AOTX_SOURCE_BOUNDARY_DATA_H\n#define AOTX_SOURCE_BOUNDARY_DATA_H\n',
           'struct aotx_source_case { const char *text; unsigned bytes, count; unsigned bounds[16]; };\n',
           'static const aotx_source_case aotx_source_cases[] = {\n']
    for data, bounds in cases:
        if len(bounds) > 16:
            raise ValueError('test boundary capacity differs')
        out.append('    {"' + ''.join('\\x%02x' % x for x in data) + '",%d,%d,{' % (len(data), len(bounds))
                   + ','.join(map(str, bounds)) + '}},\n')
    out += ['};\n#endif\n']
    (args.root / 'tests/source_boundary_data.h').write_text(''.join(out))


if __name__ == '__main__':
    main()
