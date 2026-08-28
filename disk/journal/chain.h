/* Purpose: Declare the readers of the two derived files that carry a turn and a request.
 * Owns: Nothing; each function opens and closes one file.
 * Threading: One thread; the reader writes the standard output.
 * Lifetime: The call. */
#ifndef AOTX_DISK_CHAIN_H
#define AOTX_DISK_CHAIN_H

#include "disk/wire/diskwire.h"

typedef struct aotx_chain_report {
    uint64_t lines;      /* lines read */
    uint64_t turns;      /* lines that hold a turn */
    uint64_t at;         /* the first line that breaks the chain, from 1; zero for none */
    char want[65];       /* the digest the broken line must carry */
    char got[65];        /* the digest the broken line carries */
} aotx_chain_report;

/* Prints the turns of one chain file and recomputes every digest. Returns 0 when the chain
 * holds from the first line, 1 when it breaks, and -1 when the file does not read. */
int aotx_chain_check(const char *path, aotx_chain_report *out);

/* Prints the requests of one file. Returns 0, or -1 when the file does not read. The count
 * of lines and the count of lines the reader cannot read go to the two counters. */
int aotx_requests_print(const char *path, uint64_t *lines, uint64_t *bad);

#endif
