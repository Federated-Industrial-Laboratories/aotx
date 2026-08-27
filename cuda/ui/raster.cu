/* Purpose: Compose the cell grid into the pixel buffer with the constant font.
 * Owns: The pixel buffer.
 * Launch shape: A grid of blocks; one thread for each pixel of a stride.
 * Lifetime: The whole run; the buffer is written again for every frame. */
#include "ui/ui.cuh"

__device__ unsigned int aotx_ui_pixel[AOTX_UI_PIXELS];

/* Give the color of an attribute. Three attributes give three fixed colors. */
static __device__ __forceinline__ unsigned int aotx_ui_color(unsigned int attr)
{
    if (attr == AOTX_UI_HIGH) {
        return AOTX_UI_COLOR_HIGH;
    }
    if (attr == AOTX_UI_DIM) {
        return AOTX_UI_COLOR_DIM;
    }
    return AOTX_UI_COLOR_NORMAL;
}

/* One thread takes one pixel: find the cell, read the row of the glyph, and take the bit of
 * the column. A set bit gives the color of the attribute and a clear bit gives the ground. */
__global__ void aotx_ui_raster(void)
{
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int at = blockIdx.x * blockDim.x + threadIdx.x; at < AOTX_UI_PIXELS;
         at += stride) {
        unsigned int x = at % AOTX_UI_WIDTH;
        unsigned int y = at / AOTX_UI_WIDTH;
        unsigned int cell = (y / AOTX_UI_CELL_HEIGHT) * AOTX_UI_COLS
                          + (x / AOTX_UI_CELL_WIDTH);
        unsigned int glyph = aotx_ui_grid[cell].glyph;
        if (glyph >= AOTX_UI_GLYPHS) {
            glyph = AOTX_UI_GLYPH_BOX;
        }
        unsigned int bits = aotx_ui_font[glyph][y % AOTX_UI_CELL_HEIGHT];
        unsigned int lit = (bits >> (7u - (x % AOTX_UI_CELL_WIDTH))) & 1u;
        aotx_ui_pixel[at] = lit ? aotx_ui_color(aotx_ui_grid[cell].attr) : AOTX_UI_BACK;
    }
}
