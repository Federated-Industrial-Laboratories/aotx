/* Purpose: Declare the feeder table that holds the programs of the host tools of a run.
 * Owns: Nothing; the caller holds the structure that this header declares.
 * Threading: One thread; the feeder is the only caller.
 * Lifetime: From the start of the feeder to its close.
 *
 * The device holds the catalog. The feeder holds only what it must have to run a program:
 * the module directory, the program, the timeout and the tool number. The table is a file
 * beside the requests file, so a feeder that follows a crash knows the programs of the run
 * it continues. The journal holds the imports themselves and the restore replays them to
 * the device.
 *
 * The file takes one line for every import of every kind, so the highest import number of
 * the journal stands in it. The table in memory takes a row for a tool of side host alone,
 * because the feeder runs no other kind. */
#ifndef AOTX_DISK_MODULES_H
#define AOTX_DISK_MODULES_H

#include "disk/feed/fs_tool.h"

/* Host tool modules the table holds. The figure is the module slot count of the reference
 * profile. A table that is full refuses a further row and states the figure. */
#define AOTX_MODULE_ROWS      64u
#define AOTX_MODULE_PROGRAM   128u
#define AOTX_MODULE_DIR_BYTES 512u

/* The seconds a program may run when the manifest names no timeout. */
#define AOTX_MODULE_TIMEOUT   30u

/* The name of the table file, which stands beside the requests file. */
#define AOTX_MODULE_TABLE     "modules.jsonl"

typedef struct aotx_module_row {
    uint32_t number;    /* the tool number of the module on the ring */
    uint32_t import;    /* the number of the import that installed it */
    uint32_t timeout;   /* seconds the program may run */
    uint32_t authorize; /* 1 when the operator must permit each call */
    char name[AOTX_IMPORT_NAME_BYTES];
    char dir[AOTX_MODULE_DIR_BYTES];
    char program[AOTX_MODULE_PROGRAM];
} aotx_module_row;

typedef struct aotx_modules {
    int fd;             /* the table file, open to append; -1 when the feeder has none */
    uint32_t count;
    uint32_t high;      /* the highest import number the table holds */
    uint64_t written;   /* rows appended in this run */
    uint64_t read;      /* rows read at the start */
    uint64_t refused;   /* rows the table could not hold */
    char path[AOTX_PATH_BYTES];
    aotx_module_row row[AOTX_MODULE_ROWS];
} aotx_modules;

/* Opens the table beside the requests file and reads it from its first byte. A feeder that
 * starts after a crash thus holds the programs of the run it continues. Returns 0, or -1
 * when the file does not open. A null requests path leaves the table empty and open to no
 * file, which is what a feeder with no root does. */
int aotx_modules_open(aotx_modules *m, const char *requests);

/* Appends one line for one import, and holds a row when the module is a tool of side host.
 * The kind is a word of the manifest and the side is host, device or none. The directory is
 * an absolute path. Returns 0, or -1 when the table is full or the file does not take the
 * line. */
int aotx_modules_add(aotx_modules *m, uint32_t import, const char *name, const char *kind,
                     const char *side, const char *dir, const char *program,
                     uint32_t timeout, uint32_t authorize);

/* Gives the row of a tool number, or null. */
const aotx_module_row *aotx_modules_number(const aotx_modules *m, uint32_t number);

/* Gives the row of a name, or null. */
const aotx_module_row *aotx_modules_name(const aotx_modules *m, const char *name);

/* Gives the highest import number the table holds. A feeder that starts after a crash
 * counts on from it. No two modules of one journal thus take one number, and no tool
 * number of the ring names two programs. */
uint32_t aotx_modules_high(const aotx_modules *m);

void aotx_modules_close(aotx_modules *m);

#endif
