/* Purpose: Declare the parts of the models manifest that the library and the program share.
 * Owns: Nothing; the caller owns each entry and each buffer.
 * Threading: One thread; no function here holds state between calls.
 * Lifetime: The call. */
#ifndef AOTX_MODELFILE_MANIFEST_H
#define AOTX_MODELFILE_MANIFEST_H

#include "disk/modelfile/modelfile.h"
#include "disk/wire/diskwire.h"

/* The manifest is one file in the directory that holds the model files. */
#define AOTX_MANIFEST_NAME "manifest.jsonl"

/* One line holds the file fields and the bounded turn spans, with JSON escapes. */
#define AOTX_MANIFEST_LINE 8192
#define AOTX_MANIFEST_PATH 1024

/* The largest count of entries that a program reads in one call. */
#define AOTX_MANIFEST_MAX  64

/* Joins a directory and a name. A name that starts at the root gives the name only.
 * Returns 0, or -1 when the result does not fit. */
int aotx_manifest_path(char *out, size_t out_bytes, const char *dir, const char *name);

/* Checks that a field value holds no control byte, no quotation mark, and no backslash.
 * Returns 0, or -1 when the value holds one of them. */
int aotx_manifest_field(const char *value);

/* Reads one line into an entry. Returns 0, or -1 when the line does not read. */
int aotx_manifest_line(const char *line, aotx_manifest_entry *entry);

/* Writes one entry as one line with JSON escapes and the end of line byte.
 * Returns 0, or -1 when a field is invalid or the line does not fit. */
int aotx_manifest_write_line(char *out, size_t out_bytes, const aotx_manifest_entry *entry);

#endif
