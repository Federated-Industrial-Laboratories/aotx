/* Purpose: Read bounded JSON strings and whole numbers for a model manifest.
 * Owns: No storage; cursors refer to the caller's buffers.
 * Threading: One caller for each cursor.
 * Lifetime: One parse or write call. */
#ifndef AOTX_MANIFEST_JSON_H
#define AOTX_MANIFEST_JSON_H
#include <stddef.h>
#include <stdint.h>
typedef struct aotx_manifest_json {
    const unsigned char *at;
    const unsigned char *end;
} aotx_manifest_json;
int aotx_manifest_json_take(aotx_manifest_json *j, unsigned char byte);
int aotx_manifest_json_string(aotx_manifest_json *j, unsigned char *out,
                              size_t room, size_t *length);
int aotx_manifest_json_number(aotx_manifest_json *j, uint64_t *out);
int aotx_manifest_json_end(aotx_manifest_json *j);
int aotx_manifest_json_quote(char **at, size_t *room, const unsigned char *text,
                             size_t length);
extern const char *const aotx_wrap_names[9];
#endif
