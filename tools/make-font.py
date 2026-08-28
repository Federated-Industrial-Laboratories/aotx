#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# make-font.py: write the cell font of the text grid as a constant array of glyph bitmaps.
#
# The script reads one monospaced TrueType file and draws each code point from 32 to 126 into
# an 8 by 16 pixel bitmap. One replacement glyph, a hollow box, follows them. Each row of a
# glyph is one byte, and the high bit is the left pixel.
#   make-font.py [--font PATH] [--size N] [--out PATH]
# Inputs: the TrueType file at --font.
# Outputs: the CUDA source file at --out, which holds the constant array.
# Exit codes: 0 written, 2 usage or environment error.

import argparse
import sys

FIRST = 32
LAST = 126
WIDTH = 8
HEIGHT = 16
GLYPHS = LAST - FIRST + 2
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DEFAULT_OUT = "cuda/ui/font.cu"

# The size gives an advance of 8 pixels, and the offset puts the baseline in the cell.
DEFAULT_SIZE = 13
DEFAULT_OFFSET = -1
INK = 128

BANNER = """/* Purpose: Hold the glyph bitmaps that the raster kernel reads for each cell.
 * Owns: The constant font array; {glyphs} glyphs of {height} rows of {width} bits.
 * Launch shape: Not applicable; constant memory that every thread reads.
 * Lifetime: The whole run; the array is written at load time and never changes. */

/* The script tools/make-font.py writes this file from the font file below.
 * Source font: {font}, at size {size}.
 * The license of that font file is in NOTICE.
 * Glyph 0 is code point {first} and glyph {last_index} is code point {last}.
 * Glyph {box_index} is the replacement box, which the script draws itself. */
"""


def rasterize(font_path, size, offset):
    # The import is here, so that a missing library gives one clear message.
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("make-font: the imaging library is not installed", file=sys.stderr)
        sys.exit(2)
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        print(f"make-font: cannot read the font file {font_path}", file=sys.stderr)
        sys.exit(2)
    rows = []
    for code in range(FIRST, LAST + 1):
        image = Image.new("L", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(image).text((0, offset), chr(code), font=font, fill=255)
        glyph = []
        for y in range(HEIGHT):
            bits = 0
            for x in range(WIDTH):
                if image.getpixel((x, y)) >= INK:
                    bits |= 0x80 >> x
            glyph.append(bits)
        rows.append((code, glyph))
    rows.append((0, box_glyph()))
    return rows


def box_glyph():
    # The replacement glyph is a hollow box of 6 columns and 10 rows, drawn in the cell.
    glyph = [0] * HEIGHT
    for y in range(3, 13):
        if y in (3, 12):
            glyph[y] = 0x7E
        else:
            glyph[y] = 0x42
    return glyph


def emit(rows, font_path, size, out_path):
    text = BANNER.format(glyphs=GLYPHS, height=HEIGHT, width=WIDTH, font=font_path, size=size,
                         first=FIRST, last=LAST, last_index=LAST - FIRST,
                         box_index=LAST - FIRST + 1)
    text += '#include "ui/ui.cuh"\n\n'
    text += "__constant__ unsigned char aotx_ui_font[AOTX_UI_GLYPHS][AOTX_UI_GLYPH_ROWS] = {\n"
    for index, (code, glyph) in enumerate(rows):
        body = ", ".join(f"0x{b:02x}" for b in glyph)
        text += f"    {{ {body} }},   /* glyph {index} code {code} */\n"
    text += "};\n"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = rasterize(args.font, args.size, args.offset)
    if len(rows) != GLYPHS:
        print(f"make-font: {len(rows)} glyphs, {GLYPHS} needed", file=sys.stderr)
        return 2
    blank = sum(1 for code, glyph in rows if not any(glyph))
    written = emit(rows, args.font, args.size, args.out)
    print(f"make-font: {written} glyphs to {args.out}, {blank} blank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
