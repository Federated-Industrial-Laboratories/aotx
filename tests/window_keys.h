/* Purpose: Check real window text and Enter callbacks in order.
 * Owns: The expected text and the key-pipe read buffer.
 * Launch shape: The caller draws the window; this host check sends X11 input.
 * Lifetime: One batch of distinct command lines. */
#ifndef AOTX_TEST_WINDOW_KEYS_H
#define AOTX_TEST_WINDOW_KEYS_H

#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <X11/extensions/XTest.h>
#include <X11/keysym.h>
#include <errno.h>
#include <fcntl.h>

static int aotx_test_key_read(int fd, char *text, unsigned int *length)
{
    aotx_key_body keys[64];
    ssize_t bytes;
    while ((bytes = read(fd, keys, sizeof keys)) > 0) {
        if (bytes % (ssize_t)sizeof(keys[0]) != 0) return 1;
        for (unsigned int i = 0u; i < (unsigned int)bytes / sizeof(keys[0]); ++i) {
            if (keys[i].action == GLFW_RELEASE) continue;
            unsigned int code = keys[i].key == 0u ? keys[i].codepoint : 0u;
            if (keys[i].key == GLFW_KEY_ENTER) code = '\n';
            if (code == 0u) continue;
            if (code > 127u || *length >= 1023u) return 1;
            text[(*length)++] = (char)code;
            text[*length] = '\0';
        }
    }
    return bytes < 0 && errno != EAGAIN && errno != EWOULDBLOCK;
}

static int aotx_test_key_send(Display *display, KeySym symbol)
{
    KeyCode code = XKeysymToKeycode(display, symbol);
    return code == 0 || !XTestFakeKeyEvent(display, code, True, 0)
                     || !XTestFakeKeyEvent(display, code, False, 0);
}

static int aotx_test_keys(int fd, unsigned int count)
{
    char expected[1024] = {0};
    char observed[1024] = {0};
    unsigned int length = 0u, wanted = 0u;
    int failed = 0;
    Display *display = XOpenDisplay(NULL);
    if (display == NULL || count == 0u || count > 64u) return 1;
    /* Start the fixture after any input from the window's initial frames. */
    if (aotx_test_key_read(fd, observed, &length) != 0) {
        XCloseDisplay(display);
        return 1;
    }
    printf("window: %u text bytes before the input batch\n", length);
    length = 0u;
    observed[0] = '\0';
    Window window = glfwGetX11Window(glfwGetCurrentContext());
    XSetInputFocus(display, window, RevertToParent, CurrentTime);
    for (unsigned int i = 0u; i < count && !failed; ++i) {
        char line[16];
        int size = snprintf(line, sizeof line, "note %02u", i);
        if (size <= 0 || (unsigned int)size + wanted + 1u >= sizeof expected) {
            failed = 1;
            break;
        }
        memcpy(expected + wanted, line, (size_t)size);
        wanted += (unsigned int)size;
        expected[wanted++] = '\n';
        for (int c = 0; c < size; ++c)
            failed |= aotx_test_key_send(display, (KeySym)(unsigned char)line[c]);
        failed |= aotx_test_key_send(display, XK_Return);
        XSync(display, False);
        failed |= aotx_ui_window_frame() == 0;
        failed |= aotx_test_key_read(fd, observed, &length);
    }
    for (unsigned int frame = 0u; frame < 60u && !failed; ++frame) {
        failed |= aotx_ui_window_frame() == 0;
        failed |= aotx_test_key_read(fd, observed, &length);
    }
    XCloseDisplay(display);
    failed |= length != wanted || memcmp(observed, expected, wanted) != 0;
    printf("window: %u input lines, %u of %u text bytes in callback order, %s\n",
           count, length, wanted, failed ? "failed" : "passed");
    if (failed) {
        printf("window: observed input bytes");
        for (unsigned int i = 0u; i < length; ++i) printf(" %02x", (unsigned char)observed[i]);
        printf("\n");
    }
    return failed;
}

#endif
