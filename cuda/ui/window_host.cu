/* Purpose: Show the pixel buffer in a window and send key events to the key descriptor.
 * Owns: The window, the texture, the two pixel buffer objects and their registrations.
 * Launch shape: Host glue only; the raster graph holds the kernels.
 * Lifetime: From the open before the first driver call to the close at exit. */
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "boot/check.h"
#include "ui/ui.cuh"

/* The key codes and the actions of the window library must match the values the editor
 * reads. The editor runs on the device and cannot see this library. */
static_assert(AOTX_CLI_KEY_ENTER == GLFW_KEY_ENTER, "the enter key code changed");
static_assert(AOTX_CLI_KEY_BACKSPACE == GLFW_KEY_BACKSPACE, "the backspace key code changed");
static_assert(AOTX_CLI_KEY_DELETE == GLFW_KEY_DELETE, "the delete key code changed");
static_assert(AOTX_CLI_KEY_RIGHT == GLFW_KEY_RIGHT, "the right key code changed");
static_assert(AOTX_CLI_KEY_LEFT == GLFW_KEY_LEFT, "the left key code changed");
static_assert(AOTX_CLI_KEY_DOWN == GLFW_KEY_DOWN, "the down key code changed");
static_assert(AOTX_CLI_KEY_UP == GLFW_KEY_UP, "the up key code changed");
static_assert(AOTX_CLI_KEY_HOME == GLFW_KEY_HOME, "the home key code changed");
static_assert(AOTX_CLI_KEY_END == GLFW_KEY_END, "the end key code changed");
static_assert(AOTX_CLI_KEY_KP_ENTER == GLFW_KEY_KP_ENTER, "the pad enter key code changed");
static_assert(AOTX_CLI_RELEASE == GLFW_RELEASE, "the release action changed");
static_assert(AOTX_CLI_PRESS == GLFW_PRESS, "the press action changed");
static_assert(AOTX_CLI_REPEAT == GLFW_REPEAT, "the repeat action changed");

static GLFWwindow *aotx_ui_window;
static GLuint aotx_ui_texture;
static GLuint aotx_ui_buffer[2];
static cudaGraphicsResource_t aotx_ui_shared[2];
static aotx_ui_graph aotx_ui_frame_graph;
static unsigned int aotx_ui_turn;
static int aotx_ui_keys = -1;
static unsigned int aotx_ui_errors;
static aotx_ui_frame_report aotx_ui_count;
static long long aotx_ui_last_ns;
static long long aotx_ui_first_ns;

static long long aotx_ui_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* Send one key event as a 16-byte frame. A pipe takes a write of this size whole, so a
 * frame never arrives cut in two. The bind makes the descriptor one that does not wait, so
 * a pipe that is full drops the event and the display keeps its rate. */
static void aotx_ui_send(unsigned int key, unsigned int codepoint, unsigned int action,
                         unsigned int mods)
{
    aotx_key_body body;
    if (aotx_ui_keys < 0) {
        return;
    }
    body.key = key;
    body.codepoint = codepoint;
    body.action = action;
    body.mods = mods;
    if (write(aotx_ui_keys, &body, sizeof body) != (ssize_t)sizeof body) {
        /* A pipe takes a write of this size whole or not at all, so nothing is half sent.
         * The close states the count, because a dropped key is a key the operator typed. */
        aotx_ui_count.dropped += 1ull;
    }
}

static void aotx_ui_on_key(GLFWwindow *window, int key, int code, int action, int mods)
{
    (void)window;
    (void)code;
    aotx_ui_send((unsigned int)key, 0u, (unsigned int)action, (unsigned int)mods);
}

static void aotx_ui_on_text(GLFWwindow *window, unsigned int codepoint)
{
    (void)window;
    aotx_ui_send(0u, codepoint, (unsigned int)GLFW_PRESS, 0u);
}

/* Seconds the open of the window may take. A display that makes no OpenGL window holds the
 * window library in a loop with no end. A run that has not started cannot stop at the flag
 * of a signal. The glue therefore ends the program and states what it needs. */
#define AOTX_UI_OPEN_SECONDS 30u

static void aotx_ui_late(int number)
{
    (void)number;
    static const char late[] =
        "the display did not make a window; a session that makes OpenGL windows is needed\n";
    ssize_t written = write(2, late, sizeof late - 1u);
    (void)written;
    _exit(1);
}

/* The window and its drawing context come first, so the context of the system binds to the
 * device that drives the display. */
int aotx_ui_window_open(void)
{
    signal(SIGALRM, aotx_ui_late);
    alarm(AOTX_UI_OPEN_SECONDS);
    if (glfwInit() != GLFW_TRUE) {
        fprintf(stderr, "the window library did not start\n");
        return 1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    aotx_ui_window = glfwCreateWindow((int)AOTX_UI_WIDTH, (int)AOTX_UI_HEIGHT, "AOTX-1",
                                      NULL, NULL);
    if (aotx_ui_window == NULL) {
        fprintf(stderr, "the window did not open\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(aotx_ui_window);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "the drawing library did not start\n");
        return 1;
    }
    /* The start of the drawing library leaves one error behind on some drivers. The count
     * is bounded, because a context that gives an error for every call must not hold the
     * program in a loop. */
    for (unsigned int i = 0u; i < 16u && glGetError() != GL_NO_ERROR; ++i) {
        aotx_ui_errors += 1u;
    }
    alarm(0);
    return 0;
}

int aotx_ui_window_bind(int keys_fd)
{
    if (aotx_ui_window == NULL) {
        return 1;
    }
    aotx_ui_keys = keys_fd;
    if (keys_fd >= 0) {
        /* A write that waits would hold the display at the rate of the reader. */
        int flags = fcntl(keys_fd, F_GETFL, 0);
        if (flags < 0 || fcntl(keys_fd, F_SETFL, flags | O_NONBLOCK) != 0) {
            fprintf(stderr, "the key descriptor does not take the no wait state\n");
            return 1;
        }
    }
    glfwSetKeyCallback(aotx_ui_window, aotx_ui_on_key);
    glfwSetCharCallback(aotx_ui_window, aotx_ui_on_text);

    glGenTextures(1, &aotx_ui_texture);
    glBindTexture(GL_TEXTURE_2D, aotx_ui_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)AOTX_UI_WIDTH, (GLsizei)AOTX_UI_HEIGHT,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    /* Two buffer objects: the copy fills one while the texture reads the other. The copy
     * therefore never waits for the draw of the frame before it. */
    glGenBuffers(2, aotx_ui_buffer);
    for (unsigned int i = 0u; i < 2u; ++i) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, aotx_ui_buffer[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)AOTX_UI_PIXEL_BYTES, NULL,
                     GL_STREAM_DRAW);
        aotx_check_runtime(cudaGraphicsGLRegisterBuffer(&aotx_ui_shared[i], aotx_ui_buffer[i],
                                                        cudaGraphicsRegisterFlagsWriteDiscard),
                           "cudaGraphicsGLRegisterBuffer");
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "the texture or the buffers did not open\n");
        return 1;
    }
    return aotx_ui_graph_build(&aotx_ui_frame_graph);
}

/* One frame: run the raster graph, copy the pixels into one buffer object, then draw the
 * other one. The map and the unmap synchronize, so they stay outside the graph. */
int aotx_ui_window_frame(void)
{
    void *mapped = 0;
    size_t bytes = 0;
    unsigned int now = aotx_ui_turn;
    unsigned int other = now ^ 1u;

    if (aotx_ui_window == NULL || glfwWindowShouldClose(aotx_ui_window)) {
        return 0;
    }
    glfwPollEvents();
    aotx_ui_graph_run(&aotx_ui_frame_graph);

    aotx_check_runtime(cudaGraphicsMapResources(1, &aotx_ui_shared[now], aotx_ui_frame_graph.stream),
                       "cudaGraphicsMapResources");
    aotx_check_runtime(cudaGraphicsResourceGetMappedPointer(&mapped, &bytes,
                                                            aotx_ui_shared[now]),
                       "cudaGraphicsResourceGetMappedPointer");
    aotx_check_runtime(cudaMemcpyAsync(mapped, aotx_ui_graph_pixels(),
                                       (size_t)AOTX_UI_PIXEL_BYTES, cudaMemcpyDeviceToDevice,
                                       aotx_ui_frame_graph.stream),
                       "cudaMemcpyAsync");
    aotx_check_runtime(cudaGraphicsUnmapResources(1, &aotx_ui_shared[now],
                                                  aotx_ui_frame_graph.stream),
                       "cudaGraphicsUnmapResources");
    aotx_check_runtime(cudaStreamSynchronize(aotx_ui_frame_graph.stream), "cudaStreamSynchronize");

    /* The first frame draws the buffer it filled. Every frame after it draws the other
     * buffer, so the copy of a frame never waits for the draw of the frame before it. */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,
                 aotx_ui_buffer[(aotx_ui_count.frames == 0ull) ? now : other]);
    glBindTexture(GL_TEXTURE_2D, aotx_ui_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)AOTX_UI_WIDTH, (GLsizei)AOTX_UI_HEIGHT,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glViewport(0, 0, (GLsizei)AOTX_UI_WIDTH, (GLsizei)AOTX_UI_HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, 1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, -1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, -1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glfwSwapBuffers(aotx_ui_window);
    aotx_ui_turn = other;

    long long at = aotx_ui_now_ns();
    if (aotx_ui_count.frames == 0ull) {
        aotx_ui_first_ns = at;
    } else {
        long long span = at - aotx_ui_last_ns;
        if (span < 0ll) {
            span = 0ll;
        }
        if ((unsigned long long)span > aotx_ui_count.worst_ns) {
            aotx_ui_count.worst_ns = (unsigned long long)span;
        }
        if ((unsigned long long)span > AOTX_UI_LATE_NS) {
            aotx_ui_count.late += 1ull;
        }
    }
    aotx_ui_last_ns = at;
    aotx_ui_count.frames += 1ull;
    return 1;
}

void aotx_ui_window_report(aotx_ui_frame_report *report)
{
    *report = aotx_ui_count;
    if (aotx_ui_count.frames > 1ull) {
        report->mean_ns = (unsigned long long)(aotx_ui_last_ns - aotx_ui_first_ns)
                        / (aotx_ui_count.frames - 1ull);
    }
}

int aotx_ui_window_priority(void)
{
    return aotx_ui_frame_graph.priority;
}

void aotx_ui_window_close(void)
{
    aotx_ui_graph_close(&aotx_ui_frame_graph);
    for (unsigned int i = 0u; i < 2u; ++i) {
        if (aotx_ui_shared[i] != 0) {
            cudaGraphicsUnregisterResource(aotx_ui_shared[i]);
            aotx_ui_shared[i] = 0;
        }
    }
    if (aotx_ui_buffer[0] != 0u) {
        glDeleteBuffers(2, aotx_ui_buffer);
        aotx_ui_buffer[0] = 0u;
    }
    if (aotx_ui_texture != 0u) {
        glDeleteTextures(1, &aotx_ui_texture);
        aotx_ui_texture = 0u;
    }
    if (aotx_ui_window != NULL) {
        glfwDestroyWindow(aotx_ui_window);
        aotx_ui_window = NULL;
        glfwTerminate();
    }
}
