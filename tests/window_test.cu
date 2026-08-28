/* Purpose: Open the window on the display, draw frames, and read one frame back.
 * Owns: The frame copy, the image file and the counts of the cases.
 * Launch shape: The raster graph supplies the kernels; the window draws on this thread.
 * Lifetime: One run of the test program. */
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <X11/Xlib.h>
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
        at += aotx_text_utoa(i, text + at, 64u - at);
        aotx_console_write(text, at);
        aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_FINDING, AOTX_PROV_COMPUTED, text, at,
                        0ull, 0ull, 0.0f, aotx_time_tick);
    }
}

/* Find the window of a title in the tree of the display server. */
static Window aotx_test_find(Display *display, Window at, const char *title)
{
    Window root = 0;
    Window parent = 0;
    Window *children = NULL;
    Window found = 0;
    unsigned int count = 0u;
    char *name = NULL;
    if (XFetchName(display, at, &name) != 0 && name != NULL) {
        if (strcmp(name, title) == 0) {
            found = at;
        }
        XFree(name);
    }
    if (found == 0 && XQueryTree(display, at, &root, &parent, &children, &count) != 0) {
        for (unsigned int i = 0u; i < count && found == 0; ++i) {
            found = aotx_test_find(display, children[i], title);
        }
        if (children != NULL) {
            XFree(children);
        }
    }
    return found;
}

/* Send the close request that a window manager sends: the client message WM_DELETE_WINDOW,
 * on a second connection to the display server. Nothing destroys the window; the program
 * that owns the window decides what to do. A destroy from outside takes the window away
 * from its owner and from the frame program of the desktop, which must not happen. */
static int aotx_test_request(const char *title)
{
    Display *display = XOpenDisplay(NULL);
    XEvent event;
    Window window = 0;
    if (display == NULL) {
        return 1;
    }
    window = aotx_test_find(display, DefaultRootWindow(display), title);
    if (window == 0) {
        XCloseDisplay(display);
        return 1;
    }
    memset(&event, 0, sizeof event);
    event.xclient.type = ClientMessage;
    event.xclient.window = window;
    event.xclient.message_type = XInternAtom(display, "WM_PROTOCOLS", False);
    event.xclient.format = 32;
    event.xclient.data.l[0] = (long)XInternAtom(display, "WM_DELETE_WINDOW", False);
    event.xclient.data.l[1] = CurrentTime;
    XSendEvent(display, window, False, NoEventMask, &event);
    XFlush(display);
    XCloseDisplay(display);
    return 0;
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

    /* The check is the tool as well. With --close it sends the request of a window manager
     * to the window of a title, and states nothing else. */
    if (argc > 2 && strcmp(argv[1], "--close") == 0) {
        return aotx_test_request(argv[2]);
    }
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
    /* The open of the window holds a deadline of its own and takes this one away. */
    alarm(AOTX_TEST_DEADLINE);
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

    /* The close request of a window manager reaches the window library, and the frame that
     * follows it states that the window must close. Six frames are about 100 ms at 60 Hz. */
    unsigned int after = 0u;
    aotx_test_check(aotx_test_request("AOTX-1") == 0,
                    "the close request goes to the window");
    for (unsigned int i = 0u; i < 6u; ++i) {
        if (aotx_ui_window_frame() == 0) {
            break;
        }
        after += 1u;
    }
    aotx_test_check(after < 6u, "the window stops at the close request");
    printf("window: the close request stopped the frames after %u frames\n", after);

    alarm(0);
    aotx_ui_window_close();
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    free(pixels);
    printf("window: %u cases applied, %u passed, %u failed\n", aotx_test_applied,
           aotx_test_applied - aotx_test_failed, aotx_test_failed);
    return (aotx_test_failed == 0u) ? 0 : 1;
}
