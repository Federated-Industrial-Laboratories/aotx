/* Purpose: Accept the child exit code for a CUDA illegal address.
 * Owns: Nothing.
 * Threading: The parent checks one completed child.
 * Lifetime: One fault check. */
#ifndef AOTX_TEST_FAULT_STATUS_H
#define AOTX_TEST_FAULT_STATUS_H

#include <sys/wait.h>

#define AOTX_TEST_FAULT_EXIT 7

static int aotx_test_fault_status(int status)
{
    return WIFEXITED(status) && WEXITSTATUS(status) == AOTX_TEST_FAULT_EXIT;
}

#endif
