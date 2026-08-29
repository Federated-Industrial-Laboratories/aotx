/* Purpose: Declare the feeder part that publishes one module directory as one import.
 * Owns: Nothing; the caller holds the structure that this header declares.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the first import to the close of the feeder.
 *
 * A module is a directory with a manifest and the files the manifest names. The feeder
 * reads the text and publishes it as class A records. The journal thus holds the bytes.
 * A restore builds the catalog from the journal and reads no file. */
#ifndef AOTX_DISK_IMPORT_H
#define AOTX_DISK_IMPORT_H

#include "disk/feed/fs_tool.h"

/* The bytes one file of a module may hold. The design gives the bound as a quarter of the
 * catalog arena. That figure is a build option of the device, which the disk side does not
 * see. The feeder thus applies a bound of its own of one megabyte. The bound is above the
 * arena of every build. The device applies the exact figure, and the feeder keeps out of
 * the ring only a file that no build can hold. */
#define AOTX_IMPORT_CAP  (1024u * 1024u)

/* The manifest of a module directory, and the file of a skill directory that holds none. */
#define AOTX_IMPORT_MANIFEST "module.manifest"
#define AOTX_IMPORT_SKILL    "SKILL.md"

typedef struct aotx_import {
    uint32_t number;   /* the number of the last import, from 1 */
    uint64_t imports;  /* imports published */
    uint64_t records;  /* records published, the head of each import included */
    uint64_t refusals; /* directories refused */
    uint64_t lines;    /* lines published that state a refused import */
    /* The two files under read. The buffers are large, so an import state lives beside a
     * program and not on the stack of one. */
    unsigned char bytes[AOTX_IMPORT_FILES][AOTX_IMPORT_CAP];
} aotx_import;

/* Publishes one module directory as one import. The head goes out first, and then the
 * parts of each file in order, offset by offset. The path is relative to the working
 * directory of the caller, or absolute. Nothing is executed, and three files are read at
 * most: the two files of the module and the module file of a device tool. Returns 0, or 1
 * when the directory is refused and *reason gives the cause, or -1 when the ring closed. */
int aotx_import_dir(aotx_import *s, const char *path, const aotx_inbound_ring *ring,
                    const volatile sig_atomic_t *stop, const char **reason);

/* Publishes every module directory under one directory, in name order. A refused directory
 * gives one line of the standard error and the walk goes on. One bad module must not keep
 * the other modules out of the catalog. Returns 0, or -1 when the ring closed or the stop
 * flag went to one. */
int aotx_import_tree(aotx_import *s, const char *dir, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop);

/* Reports whether one line asks for an import, and writes the path that the line names.
 * The feeder takes such a line itself: it publishes the import and not the line. */
int aotx_import_line(const unsigned char *line, uint32_t len, char *out, size_t out_bytes);

/* Imports one module directory that an operator named. The name comes from a line of the
 * standard input, or from a request that another surface made. A refusal gives one line of
 * the standard error and one input line record that states the cause. The operator then
 * sees the cause on the console. Returns 0, or -1 when the ring closed. */
int aotx_import_take(aotx_import *s, const char *path, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop);

#endif
