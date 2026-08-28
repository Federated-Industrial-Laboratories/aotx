#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# make-unicode.py: write the character class tables that the device pre-tokenizer reads.
#
#   Input: the unicodedata module of the Python that runs this script.
#   Output: cuda/text/unicode_tables.cu, or the file that the first argument names.
#   Exit codes: 0 on success, 2 when the output file cannot be written.
#
# The output file holds three range tables and the three tests that read them. The letter
# class holds the categories Lu, Ll, Lt, Lm and Lo. The number class holds the categories
# Nd, Nl and No.
#
# The space class holds the White_Space property. A regular expression engine with Unicode
# support gives that property for the space class. The property is the categories Zs, Zl
# and Zp with six control characters added, which is what the Python module gives.

import sys
import unicodedata
from pathlib import Path

LETTER = ("Lu", "Ll", "Lt", "Lm", "Lo")
NUMBER = ("Nd", "Nl", "No")
SPACE = ("Zs", "Zl", "Zp")
SPACE_CONTROL = (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85)
LAST = 0x110000
PER_LINE = 6


def ranges(test):
    # Walk the code points in order and join the ones that pass into first and last pairs.
    out = []
    start = None
    for point in range(LAST):
        if test(point):
            if start is None:
                start = point
        elif start is not None:
            out.append((start, point - 1))
            start = None
    if start is not None:
        out.append((start, LAST - 1))
    return out


def table(name, pairs):
    # Write one range table as pairs of code points, several pairs on a line.
    lines = [f"__device__ const unsigned int {name}[] = {{"]
    for at in range(0, len(pairs), PER_LINE):
        row = pairs[at:at + PER_LINE]
        text = " ".join(f"0x{first:x},0x{last:x}," for first, last in row)
        lines.append("    " + text)
    lines.append("};")
    lines.append(f"static const unsigned int {name}_pairs = {len(pairs)}u;")
    return lines


def main():
    category = unicodedata.category
    letters = ranges(lambda point: category(chr(point)) in LETTER)
    numbers = ranges(lambda point: category(chr(point)) in NUMBER)
    spaces = ranges(lambda point: category(chr(point)) in SPACE
                    or point in SPACE_CONTROL)
    version = unicodedata.unidata_version

    out = []
    out.append("/* Purpose: Give the character classes that the pre-tokenizer asks for.")
    out.append(" * Owns: The range tables of the letter, number and space classes.")
    out.append(" * Launch shape: One thread for each test; the tables hold no state.")
    out.append(" * Lifetime: The whole run. */")
    out.append('#include "text/text.cuh"')
    out.append("")
    out.append(f"/* A tool wrote this file from Unicode {version}. Do not edit it by hand.")
    out.append(" *")
    out.append(" * A table holds pairs of code points. A pair is a first and a last code")
    out.append(" * point. The pairs go up in order, so a search of two halves finds the pair")
    out.append(" * of a code point.")
    out.append(" *")
    out.append(" * The letter class is Lu, Ll, Lt, Lm and Lo. The number class is Nd, Nl and")
    out.append(" * No. The space class is the White_Space property, which is Zs, Zl and Zp")
    out.append(" * with six control characters. */")
    out.append("")
    out += table("aotx_text_letter_table", letters)
    out.append("")
    out += table("aotx_text_number_table", numbers)
    out.append("")
    out += table("aotx_text_space_table", spaces)
    out.append("")
    out.append("/* Find the pair that holds a code point. The search cuts the table in two at")
    out.append(" * each step, so the cost is the logarithm of the pair count. */")
    out.append("static __device__ int aotx_text_find(const unsigned int *pairs,")
    out.append("                                     unsigned int count, unsigned int point)")
    out.append("{")
    out.append("    unsigned int low = 0u;")
    out.append("    unsigned int high = count;")
    out.append("    while (low < high) {")
    out.append("        unsigned int mid = (low + high) >> 1;")
    out.append("        if (point < pairs[mid * 2u]) {")
    out.append("            high = mid;")
    out.append("        } else if (point > pairs[mid * 2u + 1u]) {")
    out.append("            low = mid + 1u;")
    out.append("        } else {")
    out.append("            return 1;")
    out.append("        }")
    out.append("    }")
    out.append("    return 0;")
    out.append("}")
    out.append("")
    out.append("__device__ int aotx_text_letter(unsigned int point)")
    out.append("{")
    out.append("    return aotx_text_find(aotx_text_letter_table,")
    out.append("                          aotx_text_letter_table_pairs, point);")
    out.append("}")
    out.append("")
    out.append("__device__ int aotx_text_number(unsigned int point)")
    out.append("{")
    out.append("    return aotx_text_find(aotx_text_number_table,")
    out.append("                          aotx_text_number_table_pairs, point);")
    out.append("}")
    out.append("")
    out.append("__device__ int aotx_text_space(unsigned int point)")
    out.append("{")
    out.append("    return aotx_text_find(aotx_text_space_table,")
    out.append("                          aotx_text_space_table_pairs, point);")
    out.append("}")
    out.append("")

    where = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "cuda" / "text" / "unicode_tables.cu")
    try:
        where.write_text("\n".join(out), encoding="ascii")
    except OSError as bad:
        print(f"make-unicode: {bad}", file=sys.stderr)
        return 2
    print(f"unicode {version}: letters {len(letters)} numbers {len(numbers)} "
          f"spaces {len(spaces)} lines {len(out)} file {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
