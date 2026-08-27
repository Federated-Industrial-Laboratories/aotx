/* Purpose: Check the raster and the pixel buffer path with no window on the display.
 * Owns: The device context of the display library, the pixel buffers and the case counts.
 * Launch shape: The raster graph supplies the kernels; the check runs on the host thread.
 * Lifetime: One run of the test program. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "boot/check.h"
#include "mem/mem.cuh"
#include "seam/seam.cuh"
#include "ui/ui.cuh"

#define AOTX_TEST_FRAMES  60u
#define AOTX_TEST_DEVICES 8
#define AOTX_TEST_TEXT    64u

/* Pixels of the frame that must not hold the background color, and pixels of the console
 * panel that must not hold it. A raster that draws nothing gives a frame of one color. The
 * comparison with the host figure passes on a grid of spaces, because the figure comes from
 * the same cells. The floors refuse both.
 *
 * Measured on this machine: the frame holds 7,280 lit pixels at one console line and 17,729
 * at 64. The console panel holds 467 of them and 10,885 of them. A panel that fills no cell
 * holds none. */
#define AOTX_TEST_LIT     1000u
#define AOTX_TEST_LIT_ONE 200u

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;
static EGLDisplay aotx_test_display = EGL_NO_DISPLAY;
static EGLContext aotx_test_context = EGL_NO_CONTEXT;
static EGLSurface aotx_test_surface = EGL_NO_SURFACE;
static GLuint aotx_test_buffer[2];
static cudaGraphicsResource_t aotx_test_shared[2];
static aotx_ui_cell aotx_test_grid[AOTX_UI_CELLS];
static unsigned char aotx_test_font[AOTX_UI_GLYPHS][AOTX_UI_GLYPH_ROWS];
static aotx_ui_panel aotx_test_panels[AOTX_UI_PANELS];

static void aotx_test_check(int ok, const char *what)
{
    aotx_test_applied += 1u;
    if (!ok) {
        aotx_test_failed += 1u;
        printf("raster: FAILED %s\n", what);
    }
}

/* Write console lines, so the panels have content that changes with the count. */
__global__ void aotx_test_fill(unsigned int count, unsigned int tag)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = lane; i < count; i += stride) {
        char text[AOTX_TEST_TEXT];
        unsigned int at = 0u;
        const char *head = "headless line ";
        for (unsigned int b = 0u; head[b] != '\0'; ++b) {
            text[at++] = head[b];
        }
        at += aotx_cli_utoa(tag, text + at, AOTX_TEST_TEXT - at);
        text[at++] = ' ';
        at += aotx_cli_utoa(i, text + at, AOTX_TEST_TEXT - at);
        aotx_console_write(text, at);
    }
}

/* Count the pixels that do not hold the background color. A panel of zero counts the whole
 * frame; another value counts the rectangle of that panel. */
static unsigned int aotx_test_lit(const unsigned char *bytes, const aotx_ui_panel *panel)
{
    const unsigned int *pixels = (const unsigned int *)bytes;
    unsigned int first_x = (panel == NULL) ? 0u : (unsigned int)panel->col * AOTX_UI_CELL_WIDTH;
    unsigned int first_y = (panel == NULL) ? 0u : (unsigned int)panel->row * AOTX_UI_CELL_HEIGHT;
    unsigned int last_x = (panel == NULL) ? AOTX_UI_WIDTH
                        : first_x + (unsigned int)panel->cols * AOTX_UI_CELL_WIDTH;
    unsigned int last_y = (panel == NULL) ? AOTX_UI_HEIGHT
                        : first_y + (unsigned int)panel->rows * AOTX_UI_CELL_HEIGHT;
    unsigned int count = 0u;
    for (unsigned int y = first_y; y < last_y; ++y) {
        for (unsigned int x = first_x; x < last_x; ++x) {
            if (pixels[y * AOTX_UI_WIDTH + x] != (unsigned int)AOTX_UI_BACK) {
                count += 1u;
            }
        }
    }
    return count;
}

/* Fold bytes as the other checks fold them: FNV-1a over 64 bits. */
static unsigned long long aotx_test_fold(const unsigned char *bytes, size_t count)
{
    unsigned long long hash = AOTX_FNV_BASIS;
    for (size_t i = 0u; i < count; ++i) {
        hash ^= (unsigned long long)bytes[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

/* Compose the pixels on the host from the same cells, the same font and the same rule. The
 * figure is therefore computed twice, once on each side, as the panel check computes it. */
static unsigned long long aotx_test_expect(const aotx_ui_cell *cells)
{
    unsigned int *pixels = (unsigned int *)malloc(AOTX_UI_PIXEL_BYTES);
    unsigned long long hash = 0ull;
    for (unsigned int at = 0u; at < AOTX_UI_PIXELS; ++at) {
        unsigned int x = at % AOTX_UI_WIDTH;
        unsigned int y = at / AOTX_UI_WIDTH;
        unsigned int cell = (y / AOTX_UI_CELL_HEIGHT) * AOTX_UI_COLS
                          + (x / AOTX_UI_CELL_WIDTH);
        unsigned int glyph = cells[cell].glyph;
        unsigned int attr = cells[cell].attr;
        unsigned int color = AOTX_UI_COLOR_NORMAL;
        if (glyph >= AOTX_UI_GLYPHS) {
            glyph = AOTX_UI_GLYPH_BOX;
        }
        if (attr == AOTX_UI_HIGH) {
            color = AOTX_UI_COLOR_HIGH;
        } else if (attr == AOTX_UI_DIM) {
            color = AOTX_UI_COLOR_DIM;
        }
        unsigned int bits = aotx_test_font[glyph][y % AOTX_UI_CELL_HEIGHT];
        unsigned int lit = (bits >> (7u - (x % AOTX_UI_CELL_WIDTH))) & 1u;
        pixels[at] = lit ? color : AOTX_UI_BACK;
    }
    hash = aotx_test_fold((const unsigned char *)pixels, AOTX_UI_PIXEL_BYTES);
    free(pixels);
    return hash;
}

/* Open a context on a display device. The device path of the display library needs no
 * display and no window, so a check of the raster runs where no desktop is. */
static int aotx_test_open(unsigned int *found)
{
    static const EGLint want[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    static const EGLint small[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    PFNEGLQUERYDEVICESEXTPROC devices =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC platform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDeviceEXT list[AOTX_TEST_DEVICES];
    EGLint count = 0;
    *found = 0u;
    if (devices == NULL || platform == NULL) {
        printf("raster: the display library has no device path\n");
        return 1;
    }
    if (devices(AOTX_TEST_DEVICES, list, &count) != EGL_TRUE || count <= 0) {
        printf("raster: the display library gave no device\n");
        return 1;
    }
    *found = (unsigned int)count;
    for (EGLint i = 0; i < count; ++i) {
        EGLDisplay display = platform(EGL_PLATFORM_DEVICE_EXT, list[i], NULL);
        EGLConfig config;
        EGLint configs = 0;
        EGLint major = 0;
        EGLint minor = 0;
        if (display == EGL_NO_DISPLAY || eglInitialize(display, &major, &minor) != EGL_TRUE) {
            continue;
        }
        if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE
            || eglChooseConfig(display, want, &config, 1, &configs) != EGL_TRUE
            || configs < 1) {
            eglTerminate(display);
            continue;
        }
        aotx_test_surface = eglCreatePbufferSurface(display, config, small);
        aotx_test_context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
        if (aotx_test_context == EGL_NO_CONTEXT
            || eglMakeCurrent(display, aotx_test_surface, aotx_test_surface,
                              aotx_test_context) != EGL_TRUE) {
            eglTerminate(display);
            continue;
        }
        aotx_test_display = display;
        printf("raster: device %d of %d, display library %d.%d, renderer %s\n", (int)i + 1,
               (int)count, (int)major, (int)minor, (const char *)glGetString(GL_RENDERER));
        return 0;
    }
    printf("raster: no display device gave a context\n");
    return 1;
}

/* Make the two pixel buffer objects and give them to the driver, as the window glue does. */
static int aotx_test_bind(void)
{
    glGenBuffers(2, aotx_test_buffer);
    for (unsigned int i = 0u; i < 2u; ++i) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, aotx_test_buffer[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)AOTX_UI_PIXEL_BYTES, NULL,
                     GL_STREAM_DRAW);
        if (cudaGraphicsGLRegisterBuffer(&aotx_test_shared[i], aotx_test_buffer[i],
                                         cudaGraphicsRegisterFlagsWriteDiscard)
            != cudaSuccess) {
            return 1;
        }
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return (glGetError() == GL_NO_ERROR) ? 0 : 1;
}

/* One frame: run the raster graph, then map, copy and unmap the buffer of this turn. */
static unsigned int aotx_test_frame(aotx_ui_graph *graph, unsigned int turn)
{
    void *mapped = 0;
    size_t bytes = 0;
    aotx_ui_graph_run(graph);
    aotx_check_runtime(cudaGraphicsMapResources(1, &aotx_test_shared[turn], graph->stream),
                       "cudaGraphicsMapResources");
    aotx_check_runtime(cudaGraphicsResourceGetMappedPointer(&mapped, &bytes,
                                                            aotx_test_shared[turn]),
                       "cudaGraphicsResourceGetMappedPointer");
    aotx_check_runtime(cudaMemcpyAsync(mapped, aotx_ui_graph_pixels(),
                                       (size_t)AOTX_UI_PIXEL_BYTES, cudaMemcpyDeviceToDevice,
                                       graph->stream), "cudaMemcpyAsync");
    aotx_check_runtime(cudaGraphicsUnmapResources(1, &aotx_test_shared[turn], graph->stream),
                       "cudaGraphicsUnmapResources");
    aotx_check_runtime(cudaStreamSynchronize(graph->stream), "cudaStreamSynchronize");
    return turn ^ 1u;
}

/* Draw AOTX_TEST_FRAMES frames with the content of a line count. The check then compares
 * the bytes of the pixel buffer object with the pixels of the device. It compares them with
 * the figure the host computes as well. */
static unsigned long long aotx_test_run(aotx_ui_graph *graph, unsigned int lines,
                                        unsigned int tag)
{
    unsigned char *read = (unsigned char *)malloc(AOTX_UI_PIXEL_BYTES);
    unsigned int *pixels = (unsigned int *)malloc(AOTX_UI_PIXEL_BYTES);
    unsigned int turn = 0u;
    unsigned int last = 0u;
    unsigned long long from_buffer = 0ull;
    unsigned long long from_device = 0ull;
    unsigned long long on_host = 0ull;
    unsigned int lit = 0u;
    unsigned int lit_console = 0u;

    aotx_test_fill<<<1, 32>>>(lines, tag);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int i = 0u; i < AOTX_TEST_FRAMES; ++i) {
        last = turn;
        turn = aotx_test_frame(graph, turn);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, aotx_test_buffer[last]);
    glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, (GLsizeiptr)AOTX_UI_PIXEL_BYTES, read);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    aotx_test_check(glGetError() == GL_NO_ERROR, "the pixel buffer reads back with no error");

    aotx_check_runtime(cudaMemcpyFromSymbol(pixels, aotx_ui_pixel, AOTX_UI_PIXEL_BYTES),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_grid, aotx_ui_grid,
                                            sizeof aotx_test_grid), "cudaMemcpyFromSymbol");
    from_buffer = aotx_test_fold(read, AOTX_UI_PIXEL_BYTES);
    from_device = aotx_test_fold((const unsigned char *)pixels, AOTX_UI_PIXEL_BYTES);
    on_host = aotx_test_expect(aotx_test_grid);

    lit = aotx_test_lit(read, NULL);
    lit_console = aotx_test_lit(read, &aotx_test_panels[AOTX_UI_CONSOLE]);
    aotx_test_check(from_buffer == from_device,
                    "the pixel buffer holds the pixels the raster made");
    aotx_test_check(from_buffer == on_host, "the pixels are the figure the host computes");
    /* The comparison passes on a frame of one color, because the host figure comes from the
     * same cells. The floors state that the panels put content on the frame. */
    aotx_test_check(lit >= AOTX_TEST_LIT, "the frame holds lit pixels");
    aotx_test_check(lit_console >= AOTX_TEST_LIT_ONE, "the console panel holds lit pixels");

    /* The comparison can fail: one changed pixel in the buffer gives another figure. */
    unsigned int changed = pixels[0] ^ 0x00ffffffu;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, aotx_test_buffer[last]);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, (GLsizeiptr)sizeof changed, &changed);
    glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, (GLsizeiptr)AOTX_UI_PIXEL_BYTES, read);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    aotx_test_check(aotx_test_fold(read, AOTX_UI_PIXEL_BYTES) != from_device,
                    "one changed pixel of the buffer gives another figure");
    printf("raster: %u frames at %u lines, buffer %016llx, device %016llx, host %016llx,"
           " %u lit pixels, %u of them on the console\n", AOTX_TEST_FRAMES, lines,
           from_buffer, from_device, on_host, lit, lit_console);
    free(read);
    free(pixels);
    return from_buffer;
}

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_ui_graph graph;
    unsigned long long boot_id = 0x0e91a57eull;
    unsigned int devices = 0u;

    /* The context of the display comes before the first driver call, as it does in a run
     * with a window. */
    if (aotx_test_open(&devices) != 0) {
        return 1;
    }
    glewExperimental = GL_TRUE;
    /* The drawing library reports that it found no display of the window system. The
     * pointers of the functions are there, and this path has no window system. */
    GLenum started = glewInit();
    if (started != GLEW_OK && started != GLEW_ERROR_NO_GLX_DISPLAY) {
        printf("raster: the drawing library did not start, state %u\n",
               (unsigned int)started);
        return 1;
    }
    for (unsigned int i = 0u; i < 16u && glGetError() != GL_NO_ERROR; ++i) {
        /* The start of the drawing library leaves errors behind on some drivers. */
    }
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("raster: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_font, aotx_ui_font,
                                            sizeof aotx_test_font), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_panels, aotx_ui_panel_table,
                                            sizeof aotx_test_panels), "cudaMemcpyFromSymbol");

    aotx_test_check(devices > 0u, "the display library gives a device with no display");
    aotx_test_check(aotx_test_bind() == 0, "the pixel buffers open and the driver takes them");
    aotx_test_check(aotx_ui_graph_build(&graph) == 0, "the raster graph builds");
    unsigned long long one = aotx_test_run(&graph, 1u, 1u);
    unsigned long long many = aotx_test_run(&graph, 64u, 2u);
    aotx_test_check(one != many, "the frame follows the state that the panels read");

    aotx_ui_graph_close(&graph);
    for (unsigned int i = 0u; i < 2u; ++i) {
        cudaGraphicsUnregisterResource(aotx_test_shared[i]);
    }
    glDeleteBuffers(2, aotx_test_buffer);
    eglMakeCurrent(aotx_test_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(aotx_test_display, aotx_test_surface);
    eglDestroyContext(aotx_test_display, aotx_test_context);
    eglTerminate(aotx_test_display);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("raster: %u cases applied, %u passed, %u failed\n", aotx_test_applied,
           aotx_test_applied - aotx_test_failed, aotx_test_failed);
    if (aotx_test_applied == 0u) {
        printf("raster: no case ran\n");
        return 1;
    }
    return (aotx_test_failed == 0u) ? 0 : 1;
}
