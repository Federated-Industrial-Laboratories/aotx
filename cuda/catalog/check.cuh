/* Purpose: Declare the checks that judge one module against the tool contract.
 * Owns: Nothing; the verdict is a device global of the check and the caller holds the rest.
 * Launch shape: One thread for each request row; the import and the report take one thread.
 * Lifetime: One run of the check program. */
#ifndef AOTX_CATALOG_CHECK_CUH
#define AOTX_CATALOG_CHECK_CUH

#include "catalog/catalog.cuh"
#include "tool/module.cuh"

/* Bytes of one text the report gives the host: a reason, the example line, the program. */
#define AOTX_CHECK_TEXT_BYTES 256u

/* What the batch of one run gave. Every figure comes from a kernel, so the check program
 * compares nothing on the host. */
typedef struct aotx_check_verdict {
    unsigned int rows;        /* rows the batch held */
    unsigned int taken;       /* rows the fill marked for the module */
    unsigned int done;        /* taken rows the module finished */
    unsigned int status_bad;  /* rows with a status the contract does not allow */
    unsigned int over;        /* rows whose length went past the bound */
    unsigned int untaken;     /* untaken rows the module wrote */
    unsigned int empty;       /* taken rows with a result of no length */
    unsigned int longest;     /* the longest result of the run */
} aotx_check_verdict;

extern __device__ aotx_check_verdict aotx_check_out;

/* What the catalog holds for one entry after the import. */
typedef struct aotx_check_entry {
    unsigned int state;       /* AOTX_CATALOG_* */
    unsigned int why;         /* AOTX_CATALOG_WHY_* of a refused entry */
    unsigned int figure;
    unsigned int side;        /* AOTX_CATALOG_SIDE_* */
    unsigned int arguments;
    unsigned int timeout;
    unsigned int deadline;
    unsigned int authorize;
    unsigned int example_len;
    char reason[AOTX_CHECK_TEXT_BYTES];
    char example[AOTX_CHECK_TEXT_BYTES];
    char program[AOTX_CHECK_TEXT_BYTES];
} aotx_check_entry;

/* Import one manifest through the reader the apply uses. The bytes stand in device memory
 * and the name and the path are device texts that end with a zero byte. */
__global__ void aotx_check_import(const unsigned char *bytes, unsigned int length,
                                  unsigned int kind, const char *name, const char *path);

/* Put the digest of the module file in the entry, as the head of an import does. */
__global__ void aotx_check_digest(unsigned int entry, const unsigned char *digest);

/* Read one entry back for the report of the check program. */
__global__ void aotx_check_report(unsigned int entry, aotx_check_entry *out);

/* Fill the batch of one module node: rows taken, every other row a canary. */
__global__ void aotx_check_fill(unsigned int node, unsigned int entry, unsigned int rows);

/* Judge what the module wrote: the done word, the status, the length and the canary. */
__global__ void aotx_check_judge(unsigned int rows);

#endif
