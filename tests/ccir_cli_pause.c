/* Purpose: Pause the real CLI before its writer open for an interleaving check.
 * Owns: An open interceptor and two inherited synchronization pipes.
 * Threading: One CLI process; the parent releases it after another writer commits.
 * Lifetime: The preload module is present only in the instrumented child. */
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int open(const char *path, int flags, ...)
{
    int (*real_open)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
    const char *target = getenv("AOTX_CCIR_PAUSE_PATH");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags); mode = (mode_t)va_arg(args, int); va_end(args);
    }
    if (!real_open) _exit(90);
    if (target && !strcmp(path, target) && (flags & O_ACCMODE) == O_RDWR) {
        const char *ready_text = getenv("AOTX_CCIR_PAUSE_READY");
        const char *go_text = getenv("AOTX_CCIR_PAUSE_GO");
        struct pollfd wait;
        char byte = 'r';
        int ready;
        if (!ready_text || !go_text) _exit(90);
        ready = atoi(ready_text); wait.fd = atoi(go_text); wait.events = POLLIN; wait.revents = 0;
        if (write(ready, &byte, 1u) != 1 || poll(&wait, 1u, 10000) != 1 ||
            read(wait.fd, &byte, 1u) != 1 || byte != 'g') _exit(90);
    }
    return real_open(path, flags, mode);
}
