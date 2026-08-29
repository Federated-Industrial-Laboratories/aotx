/* Purpose: Declare the feeder part that runs a host tool as a child program.
 * Owns: Nothing; the caller holds the structure that this header declares.
 * Threading: One thread; the feeder starts and reaps every child from its poll loop.
 * Lifetime: From the first program to the close of the feeder.
 *
 * A host tool of the catalog is a program in any language. The feeder gives the child the
 * module directory as its working directory. The requests line goes on the standard input.
 * The standard output is the reply content under the cap. The standard error is the reason
 * of an exit status that is not zero. The built-in tool that runs a command line takes the
 * same path with the allowed root as the working directory. */
#ifndef AOTX_DISK_RUN_TOOL_H
#define AOTX_DISK_RUN_TOOL_H

#include "disk/feed/fs_tool.h"

/* The programs the feeder runs at one time. The figure is the request slot count, so every
 * request of one tick can hold a child of its own. */
#define AOTX_TOOL_PROGRAMS_MAX 64u

/* The bytes of the standard error that a reason may state. */
#define AOTX_TOOL_ERR_BYTES 256u

/* The command interpreter of the built-in tool that runs a command line. */
#define AOTX_RUN_SHELL "/bin/sh"

typedef struct aotx_child {
    int used;
    int pid;
    int out_fd;           /* the read end of the standard output, or -1 at its end */
    int err_fd;           /* the read end of the standard error, or -1 at its end */
    int killed;           /* one when the timeout ended the program */
    int reaped;           /* one when the exit status is in hand */
    int status;           /* the exit status, or the negative of the signal number */
    uint32_t agent;
    uint32_t request;
    uint32_t timeout;     /* seconds the program may run */
    uint64_t deadline_ns; /* the monotonic time the program must end before */
    uint32_t out_len;
    int out_over;         /* one when the output is longer than the cap */
    uint32_t err_len;
    char err[AOTX_TOOL_ERR_BYTES];
    unsigned char out[AOTX_FS_CAP];
} aotx_child;

typedef struct aotx_children {
    uint64_t started;
    uint64_t ended;
    uint64_t killed;   /* programs the timeout ended */
    uint64_t refused;  /* requests that found no free place */
    aotx_child at[AOTX_TOOL_PROGRAMS_MAX];
} aotx_children;

/* Starts one program. The working directory of the child is dir_fd, which stays open for
 * the caller. The file is the program, and argv is its argument list with the file as its
 * first member and a null last member. The line goes on the standard input of the child.
 * The environment of the child is the environment of the feeder with the request, the
 * agent and the tool added. Returns 0, or 1 with the reason when the program does not
 * start. */
int aotx_run_start(aotx_children *c, uint32_t agent, uint32_t request, int dir_fd,
                   const char *file, char *const argv[], const char *tool, const char *line,
                   uint32_t timeout, const char **reason);

/* Takes the output of every program that runs, ends a program that passed its timeout, and
 * publishes the reply of every program that ended. Returns 0, or -1 when the ring closed
 * or the stop flag went to one. */
int aotx_run_poll(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                  const volatile sig_atomic_t *stop);

/* Ends every program that still runs and reaps it. The feeder calls this at its close, so
 * no program of the run stays behind it. */
void aotx_run_close(aotx_children *c);

#endif
