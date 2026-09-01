/* Purpose: Declare the catalog, the model store, and the fetch operation.
 * Owns: Nothing; callers own all tables and text buffers.
 * Threading: One thread for each store operation.
 * Lifetime: One call, except that returned tables stay valid in caller storage. */
#ifndef AOTX_DISK_MODELS_H
#define AOTX_DISK_MODELS_H

#include <stddef.h>
#include <stdint.h>

#include "disk/modelfile/modelfile.h"

#define AOTX_MODEL_CATALOG_MAX 128u
#define AOTX_MODEL_LINE        2048u
#define AOTX_MODEL_PATH        1024u
#define AOTX_MODEL_NAME        96u
#define AOTX_MODEL_TEXT        256u
#define AOTX_MODEL_FETCH_TIMEOUT 30u

typedef struct aotx_model_catalog_entry {
    char name[AOTX_MODEL_NAME];
    char role[32];
    char repository[160];
    char file[AOTX_MODEL_TEXT];
    char revision[64];
    uint64_t bytes;
    char sha256[AOTX_SHA256_HEX];
    char license[64];
    char quant[32];
    char profiles[64];
    int verified;
    char source[AOTX_MODEL_TEXT];
    char note[AOTX_MODEL_TEXT];
} aotx_model_catalog_entry;

typedef struct aotx_model_catalog {
    aotx_model_catalog_entry entry[AOTX_MODEL_CATALOG_MAX];
    unsigned int count;
} aotx_model_catalog;

typedef struct aotx_model_store_record {
    char name[AOTX_MODEL_NAME];
    char file[AOTX_MODEL_TEXT];
    uint64_t bytes;
    char sha256[AOTX_SHA256_HEX];
    char source[AOTX_MODEL_TEXT * 2u];
    char date[32];
    char revision[64];
    int verified;
} aotx_model_store_record;

enum aotx_model_state {
    AOTX_MODEL_NOT_FETCHED = 0,
    AOTX_MODEL_ON_DISK,
    AOTX_MODEL_NOT_ACTIVE,
    AOTX_MODEL_FETCHING,
    AOTX_MODEL_DIGEST_DIFFERS
};

typedef struct aotx_model_view {
    aotx_model_catalog_entry catalog;
    enum aotx_model_state state;
    uint64_t bytes_on_disk;
    int verified;
} aotx_model_view;

int aotx_model_catalog_read(const char *path, aotx_model_catalog *catalog,
                            char *reason, size_t reason_bytes);
const aotx_model_catalog_entry *aotx_model_catalog_find(const aotx_model_catalog *catalog,
                                                        const char *name);

int aotx_model_store_line(const char *line, aotx_model_store_record *record);
int aotx_model_store_write_line(char *out, size_t bytes,
                                const aotx_model_store_record *record);
int aotx_model_store_read(const char *dir, aotx_model_store_record *records,
                          unsigned int most);
int aotx_model_store_append(const char *dir, const aotx_model_store_record *record,
                            char *reason, size_t reason_bytes);

int aotx_model_store_scan(const char *dir, const aotx_model_catalog *catalog,
                          aotx_model_view *views, unsigned int most,
                          char *reason, size_t reason_bytes);
const char *aotx_model_state_text(enum aotx_model_state state);
int aotx_model_store_check(const char *dir, char *reason, size_t reason_bytes);
int aotx_model_store_activate(const char *dir, const aotx_model_catalog_entry *entry,
                              const char *role, char *reason, size_t reason_bytes);
int aotx_model_store_remove(const char *dir, const aotx_model_catalog_entry *entry,
                            char *reason, size_t reason_bytes);

/* Write declared sampling controls from on-disk model metadata. */
int aotx_model_parameters_line(const char *path, const char *name, char *out, size_t bytes);
int aotx_model_parameters_scan(const char *dir, const aotx_model_catalog *catalog,
                               char *reason, size_t reason_bytes);

int aotx_model_fetch(const char *dir, const aotx_model_catalog_entry *entry,
                     unsigned int connect_timeout,
                     char *reason, size_t reason_bytes);

#endif
