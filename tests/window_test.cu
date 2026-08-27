/* Purpose: Open the window on the display, draw frames, and read one frame back.
 * Owns: The frame copy, the image file and the counts of the cases.
 * Launch shape: The raster graph supplies the kernels; the window draws on this thread.
 * Lifetime: One run of the test program. */
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>

#include "boot/check.h"
#include "bus/bus.cuh"
#include "mem/mem.cuh"
#include "seam/seam.cuh"
#include "ui/ui.cuh"

#define AOTX_TEST_FRAMES  60u
#define AOTX_TEST_COLORS  32u
#define AOTX_TEST_LIT     1000u

/* Seconds the check gives the display. A display that makes no window holds the window
 * library in a loop. The check therefore ends itself and states what it needs. */
#define AOTX_TEST_DEADLINE 45u

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;

static void aotx_test_late(int number)
{
    (void)number;
    static const char late[] =
        "window: the display did not make a window; a session that makes OpenGL windows is "
        "needed\n";
    ssize_t written = write(2, late, sizeof late - 1u);
    (void)written;
    _exit(1);
}

static void aotx_test_check(int ok, const char *what)
{
    aotx_test_applied += 1u;
    if (!ok) {
        aotx_test_failed += 1u;
        printf("window: FAILED %s\n", what);
    }
}

/* Fill the console and the bus with distinct lines, so the panels have content to show. */
__global__ void aotx_test_fill(unsigned int count)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = lane; i < count; i += stride) {
        char text[64];
        unsigned int at = 0u;
        const char *head = "the window shows line ";
        for (unsigned int b = 0u; head[b] != '\0'; ++b) {
            text[at++] = head[b];
        }
        at += aotx_cli_utoa(i, text + at, 64u - at);
        aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_CONSOLE, 0u, text, at);
        aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_FINDING, AOTX_PROV_COMPUTED, text, at,
                        0ull, 0ull, 0.0f, aotx_time_tick);
    }
}

/* Write the frame as an image file. The rows come from the display bottom first, so the
 * file takes them in the other order. */
static int aotx_test_write(const char *path, const unsigned char *pixels)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return 1;
    }
    fprintf(file, "P6\n%u %u\n255\n", AOTX_UI_WIDTH, AOTX_UI_HEIGHT);
    for (unsigned int y = 0u; y < AOTX_UI_HEIGHT; ++y) {
        const unsigned char *row = pixels + (size_t)(AOTX_UI_HEIGHT - 1u - y)
                                   * AOTX_UI_WIDTH * 3u;
        fwrite(row, 1u, (size_t)AOTX_UI_WIDTH * 3u, file);
    }
    fclose(file);
    return 0;
}

int main(int argc, char **argv)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    unsigned long long boot_id = 0x0117d0c5ull;
    const char *path = (argc > 1) ? argv[1] : "window.ppm";
    unsigned char *pixels = NULL;
    unsigned int colors[AOTX_TEST_COLORS];
    unsigned int distinct = 0u;
    unsigned int lit = 0u;
    unsigned int frames = 0u;

    signal(SIGALRM, aotx_test_late);
    alarm(AOTX_TEST_DEADLINE);
    if (getenv("DISPLAY") == NULL && getenv("WAYLAND_DISPLAY") == NULL) {
        printf("window: this check needs a display; DISPLAY is not set\n");
        return 1;
    }
    /* The window and its drawing context come before the first driver call. */
    if (aotx_ui_window_open() != 0) {
        printf("window: the display did not give a window\n");
        return 1;
    }
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("window: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_test_fill<<<2, 32>>>(40u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_test_check(aotx_ui_window_bind(-1) == 0, "the texture and the buffers open");
    for (unsigned int i = 0u; i < AOTX_TEST_FRAMES; ++i) {
        if (aotx_ui_window_frame() == 0) {
            break;
        }
        frames += 1u;
    }
    aotx_test_check(frames == AOTX_TEST_FRAMES, "the window drew every frame");

    pixels = (unsigned char *)malloc((size_t)AOTX_UI_WIDTH * AOTX_UI_HEIGHT * 3u);
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, (GLsizei)AOTX_UI_WIDTH, (GLsizei)AOTX_UI_HEIGHT, GL_RGB,
                 GL_UNSIGNED_BYTE, pixels);
    aotx_test_check(glGetError() == GL_NO_ERROR, "the frame reads back with no error");

    for (size_t i = 0u; i < (size_t)AOTX_UI_WIDTH * AOTX_UI_HEIGHT; ++i) {
        unsigned int color = ((unsigned int)pixels[i * 3u] << 16)
                           | ((unsigned int)pixels[i * 3u + 1u] << 8)
                           | (unsigned int)pixels[i * 3u + 2u];
        unsigned int seen = 0u;
        if (color != 0x0b0d10u) {
            lit += 1u;
        }
        for (unsigned int c = 0u; c < distinct; ++c) {
            if (colors[c] == color) {
                seen = 1u;
                break;
            }
        }
        if (seen == 0u && distinct < AOTX_TEST_COLORS) {
            colors[distinct] = color;
            distinct += 1u;
        }
    }
    aotx_test_check(aotx_test_write(path, pixels) == 0, "the frame goes to the image file");
    aotx_test_check(distinct >= 2u, "the frame holds two colors or more");
    aotx_test_check(lit >= AOTX_TEST_LIT, "the frame holds a thousand lit pixels or more");
    printf("window: %u frames, %u colors, %u lit pixels, image %s\n", frames, distinct, lit,
           path);
    for (unsigned int c = 0u; c < distinct && c < 8u; ++c) {
        printf("window: color %u is %06x\n", c, colors[c]);
    }

    alarm(0);
    aotx_ui_window_close();
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    free(pixels);
    printf("window: %u cases applied, %u passed, %u failed\n", aotx_test_applied,
           aotx_test_applied - aotx_test_failed, aotx_test_failed);
    return (aotx_test_failed == 0u) ? 0 : 1;
}
