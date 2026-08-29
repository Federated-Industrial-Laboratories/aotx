#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# make-splash.py: write the splash art of the terminal program as text files of cells.
#
# The script reads one image and writes two text files for each size of a fixed list. The
# braille file holds one code point of the braille block for each cell, which is 2 by 4
# dots. The ascii file holds one character of a ramp for each cell.
#
# The image keeps its aspect and stands in the middle of the cell grid. Each line of an
# output file holds exactly the columns the file name says, so a file of the wrong width
# is refused.
#   make-splash.py [--image PATH] [--out DIR]
# Inputs: the image at --image.
# Outputs: two text files under --out for each size, named by the columns, the rows and
#   the form (<cols>x<rows>.braille and <cols>x<rows>.ascii).
# Exit codes: 0 written, 2 usage or environment error.

import argparse
import sys

DEFAULT_IMAGE = "share/splash/mark.png"
DEFAULT_OUT = "share/splash"

# The sizes the program can show. The program picks the largest one that fits and never
# scales art at run time. This list is thus the whole set of sizes that exist.
SIZES = ((160, 40), (120, 30), (80, 20), (60, 12), (40, 8))

# The first code point of the braille block. The dot bits follow the order of the block.
# The left column holds the bits 0, 1, 2 and 6 from the top. The right column holds the
# bits 3, 4, 5 and 7.
BRAILLE_FIRST = 0x2800
BRAILLE_BITS = ((0, 3), (1, 4), (2, 5), (6, 7))

# The ramp runs from no ink to full ink. A terminal shows the mark on a dark ground. A
# bright cell of the mark thus takes the character with the most ink.
RAMP = " .:-=+*#%@"

# A cell of the text grid is 8 pixels wide and 16 pixels tall (tools/make-font.py). A cell
# is thus twice as tall as it is wide. The ascii art gives the image that many more
# columns than rows to keep its aspect. The braille art needs no such figure, because 2
# by 4 dots in that cell are square.
CELL_TALLNESS = 2

# The mark is bright strokes on a dark ground, and most of the image is near black. This
# power opens the dark half of the range. The strokes then hold enough ink to make a
# shape at 80 columns, and the ground stays empty. Without the lift the art is scattered
# dots.
LIFT = 0.6


def load_luminance(path):
    """Give the image as a list of rows of luminance, 0 to 255, over a black ground."""
    # The import is here, so that a missing library gives one clear message.
    try:
        from PIL import Image
    except ImportError:
        print("make-splash: the imaging library is not installed", file=sys.stderr)
        sys.exit(2)
    try:
        image = Image.open(path).convert("RGBA")
    except OSError:
        print(f"make-splash: cannot read the image {path}", file=sys.stderr)
        sys.exit(2)
    ground = Image.new("RGBA", image.size, (0, 0, 0, 255))
    flat = Image.alpha_composite(ground, image).convert("L")
    return flat.point(lambda level: int(255.0 * ((level / 255.0) ** LIFT)))


def fit(image, width, height, tallness):
    """Scale the image into the box, aspect kept, and give the scaled image and its box."""
    from PIL import Image

    source_width, source_height = image.size
    # A cell that is `tallness` times as tall as it is wide needs that many more columns
    # than rows to show the same shape.
    aspect = (source_width / source_height) * tallness
    if width / height >= aspect:
        rows = height
        columns = max(1, round(height * aspect))
    else:
        columns = width
        rows = max(1, round(width / aspect))
    # The area filter takes the mean of the pixels of each cell. A filter that sharpens
    # gives the error diffusion an edge that is not in the image.
    return image.resize((columns, rows), Image.BOX)


def dither(image, width, height):
    """Give a bitmap of the box, one dot for each bright pixel of the scaled image.

    The error diffusion runs over the inverted luminance, so a bright stroke of the mark
    becomes ink, and ink becomes a raised dot on a dark terminal. The weights are the ones
    of Floyd and Steinberg: 7/16 to the right, and 3/16, 5/16 and 1/16 to the row below.
    """
    art = image.load()
    columns, rows = image.size
    left = (width - columns) // 2
    top = (height - rows) // 2
    value = [[255.0 - art[x, y] for x in range(columns)] for y in range(rows)]
    dots = [[0] * width for _ in range(height)]
    for y in range(rows):
        for x in range(columns):
            old = value[y][x]
            new = 0.0 if old < 128.0 else 255.0
            value[y][x] = new
            if new == 0.0:
                dots[top + y][left + x] = 1
            error = old - new
            if x + 1 < columns:
                value[y][x + 1] += error * 7.0 / 16.0
            if y + 1 < rows:
                if x > 0:
                    value[y + 1][x - 1] += error * 3.0 / 16.0
                value[y + 1][x] += error * 5.0 / 16.0
                if x + 1 < columns:
                    value[y + 1][x + 1] += error * 1.0 / 16.0
    return dots


def braille_lines(image, columns, rows):
    """Give the braille art: one line for each row of cells, each of `columns` cells."""
    dots = dither(image, columns * 2, rows * 4)
    lines = []
    for row in range(rows):
        line = []
        for column in range(columns):
            bits = 0
            for dot_row in range(4):
                for dot_column in range(2):
                    if dots[row * 4 + dot_row][column * 2 + dot_column]:
                        bits |= 1 << BRAILLE_BITS[dot_row][dot_column]
            line.append(chr(BRAILLE_FIRST + bits))
        lines.append("".join(line))
    return lines


def ascii_lines(image, columns, rows):
    """Give the ascii art: one line for each row of cells, each of `columns` cells."""
    art = image.load()
    art_columns, art_rows = image.size
    left = (columns - art_columns) // 2
    top = (rows - art_rows) // 2
    lines = []
    for row in range(rows):
        line = []
        for column in range(columns):
            x = column - left
            y = row - top
            if 0 <= x < art_columns and 0 <= y < art_rows:
                step = art[x, y] * (len(RAMP) - 1) // 255
                line.append(RAMP[step])
            else:
                line.append(RAMP[0])
        lines.append("".join(line))
    return lines


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as out:
        for line in lines:
            out.write(line)
            out.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Write the splash art as text files.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    image = load_luminance(args.image)
    for columns, rows in SIZES:
        dot_art = fit(image, columns * 2, rows * 4, 1)
        cell_art = fit(image, columns, rows, CELL_TALLNESS)
        name = f"{columns}x{rows}"
        write_lines(f"{args.out}/{name}.braille", braille_lines(dot_art, columns, rows))
        write_lines(f"{args.out}/{name}.ascii", ascii_lines(cell_art, columns, rows))
        print(f"make-splash: {name} braille {dot_art.size[0]}x{dot_art.size[1]} dots,"
              f" ascii {cell_art.size[0]}x{cell_art.size[1]} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
