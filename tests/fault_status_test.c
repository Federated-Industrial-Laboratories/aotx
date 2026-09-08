/* Purpose: Reject child failures that do not report a CUDA illegal address.
 * Owns: The child process fixtures.
 * Threading: The parent starts and waits for each batch of children.
 * Lifetime: One test run. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "fault_status.h"

static void aotx_test_child(unsigned int mode)
{
    if (mode == 0u) _exit(AOTX_TEST_FAULT_EXIT);
    if (mode == 1u) _exit(0);
    if (mode == 2u) _exit(2);
    if (mode == 3u) {
        char *argv[] = { (char *)"/dev/null/aotx-test", NULL };
        execv(argv[0], argv);
        _exit(127);
    }
    signal(SIGTERM, SIG_DFL);
    raise(SIGTERM);
    _exit(3);
}

int main(void)
{
    const unsigned int counts[] = { 1u, 64u };
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    for (unsigned int c = 0u; c < 2u; ++c) {
        for (unsigned int mode = 0u; mode < 5u; ++mode) {
            pid_t children[64];
            for (unsigned int i = 0u; i < counts[c]; ++i) {
                children[i] = fork();
                if (children[i] == 0) aotx_test_child((mode + i) % 5u);
            }
            for (unsigned int i = 0u; i < counts[c]; ++i) {
                int status = 0;
                pid_t result = -1;
                applied += 1u;
                if (children[i] > 0) {
                    do {
                        result = waitpid(children[i], &status, 0);
                    } while (result < 0 && errno == EINTR);
                }
                int expected = (mode + i) % 5u == 0u;
                if (result <= 0 || aotx_test_fault_status(status) != expected) {
                    printf("fault_status: child %u in mode %u has an incorrect result\n", i, mode);
                    failed += 1u;
                }
            }
        }
    }
    printf("fault_status: %u cases applied, %u failed\n", applied, failed);
    return applied == 325u && failed == 0u ? 0 : 1;
}
