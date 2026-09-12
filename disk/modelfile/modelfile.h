/* Purpose: Read a GGUF model file: its metadata, its tensor table, and its tensor bytes.
 * Owns: The tensor table and the metadata of one open file.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: From open to close. */
#ifndef AOTX_MODELFILE_H
#define AOTX_MODELFILE_H

#include <stddef.h>
#include <stdint.h>
#include "disk/modelfile/wrap.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The file format, as read from the file and checked at open. The magic is the four bytes
 * "GGUF". The version is 3. The header holds the tensor count and the metadata count. Then
 * come the metadata pairs, the tensor infos, padding to the alignment, and the tensor bytes.
 *
 * A tensor info holds a name, a dimension count, the dimensions, a type, and a byte offset
 * from the start of the tensor bytes. The alignment comes from general.alignment, or 32. */
#define AOTX_GGUF_MAGIC        0x46554747u   /* "GGUF" in little-endian byte order */
#define AOTX_GGUF_VERSION      3u
#define AOTX_GGUF_ALIGN        32u

/* Metadata value types, by the number the file stores. */
#define AOTX_GGUF_U8           0u
#define AOTX_GGUF_I8           1u
#define AOTX_GGUF_U16          2u
#define AOTX_GGUF_I16          3u
#define AOTX_GGUF_U32          4u
#define AOTX_GGUF_I32          5u
#define AOTX_GGUF_F32          6u
#define AOTX_GGUF_BOOL         7u
#define AOTX_GGUF_STRING       8u
#define AOTX_GGUF_ARRAY        9u
#define AOTX_GGUF_U64          10u
#define AOTX_GGUF_I64          11u
#define AOTX_GGUF_F64          12u

/* Tensor types the system reads, by the number the file stores. Other types open but do
 * not stream. */
#define AOTX_TENSOR_F32        0u
#define AOTX_TENSOR_F16        1u
#define AOTX_TENSOR_Q4_0       2u
#define AOTX_TENSOR_Q4_1       3u
#define AOTX_TENSOR_Q5_0       6u
#define AOTX_TENSOR_Q5_1       7u
#define AOTX_TENSOR_Q8_0       8u
#define AOTX_TENSOR_Q2_K       10u
#define AOTX_TENSOR_Q3_K       11u
#define AOTX_TENSOR_Q4_K       12u
#define AOTX_TENSOR_Q5_K       13u
#define AOTX_TENSOR_Q6_K       14u

/* Legacy blocks hold 32 weights. K super blocks hold 256 weights.
 * Each type table row gives its exact block width and byte count. */
#define AOTX_BLOCK_WEIGHTS     32u
#define AOTX_Q4_0_BYTES        18u
#define AOTX_Q4_1_BYTES        20u
#define AOTX_Q5_0_BYTES        22u
#define AOTX_Q5_1_BYTES        24u
#define AOTX_Q8_0_BYTES        34u
#define AOTX_SUPER_WEIGHTS     256u
#define AOTX_Q2_K_BYTES        84u
#define AOTX_Q3_K_BYTES        110u
#define AOTX_Q4_K_BYTES        144u
#define AOTX_Q5_K_BYTES        176u
#define AOTX_Q6_K_BYTES        210u

/* The reader and the weight loader use this same block type list. */
#define AOTX_TENSOR_TYPE_TABLE(X) \
    X(AOTX_TENSOR_F32, "F32", 1u, 4u) \
    X(AOTX_TENSOR_F16, "F16", 1u, 2u) \
    X(AOTX_TENSOR_Q4_0, "Q4_0", AOTX_BLOCK_WEIGHTS, AOTX_Q4_0_BYTES) \
    X(AOTX_TENSOR_Q4_1, "Q4_1", AOTX_BLOCK_WEIGHTS, AOTX_Q4_1_BYTES) \
    X(AOTX_TENSOR_Q5_0, "Q5_0", AOTX_BLOCK_WEIGHTS, AOTX_Q5_0_BYTES) \
    X(AOTX_TENSOR_Q5_1, "Q5_1", AOTX_BLOCK_WEIGHTS, AOTX_Q5_1_BYTES) \
    X(AOTX_TENSOR_Q8_0, "Q8_0", AOTX_BLOCK_WEIGHTS, AOTX_Q8_0_BYTES) \
    X(AOTX_TENSOR_Q2_K, "Q2_K", AOTX_SUPER_WEIGHTS, AOTX_Q2_K_BYTES) \
    X(AOTX_TENSOR_Q3_K, "Q3_K", AOTX_SUPER_WEIGHTS, AOTX_Q3_K_BYTES) \
    X(AOTX_TENSOR_Q4_K, "Q4_K", AOTX_SUPER_WEIGHTS, AOTX_Q4_K_BYTES) \
    X(AOTX_TENSOR_Q5_K, "Q5_K", AOTX_SUPER_WEIGHTS, AOTX_Q5_K_BYTES) \
    X(AOTX_TENSOR_Q6_K, "Q6_K", AOTX_SUPER_WEIGHTS, AOTX_Q6_K_BYTES)

static inline const char *aotx_tensor_type_name(uint32_t type)
{
#define AOTX_TENSOR_NAME_CASE(value, name, block, bytes) case value: return name;
    switch (type) {
        AOTX_TENSOR_TYPE_TABLE(AOTX_TENSOR_NAME_CASE)
    default: return NULL;
    }
#undef AOTX_TENSOR_NAME_CASE
}

#define AOTX_TENSOR_NAME_BYTES 128u
#define AOTX_TENSOR_DIMS       4u

typedef struct aotx_tensor_info {
    char     name[AOTX_TENSOR_NAME_BYTES];
    uint32_t dim_count;
    uint32_t type;
    uint64_t dims[AOTX_TENSOR_DIMS];
    uint64_t offset;            /* from the start of the tensor bytes */
    uint64_t bytes;             /* computed from the dimensions and the type */
} aotx_tensor_info;

/* A string array from the metadata, kept as one byte run with an offset table. Token
 * strings and merge strings arrive this way and cross to the device unchanged. */
typedef struct aotx_string_array {
    uint64_t count;
    const uint8_t *bytes;       /* every string, one after the other, no separators */
    const uint64_t *offsets;    /* count + 1 entries; string i is [offsets[i], offsets[i+1]) */
} aotx_string_array;

typedef struct aotx_modelfile aotx_modelfile;

/* Open the file, read and check the header, the metadata and the tensor table.
 * Returns 0 on success; 1 when the file cannot be read; 2 when the format is wrong. */
int aotx_modelfile_open(const char *path, aotx_modelfile **file);
/* Duplicate a leased descriptor and read one bounded subfile, including its tensors.
 * The caller retains its descriptor. Close releases the duplicate on success or failure. */
int aotx_modelfile_open_extent(const char *name, int fd, uint64_t offset,
                               uint64_t bytes, aotx_modelfile **file);
void aotx_modelfile_close(aotx_modelfile *file);

/* Read a header through a bounded byte source. The callback fills exactly bytes bytes
 * at offset, or returns a nonzero error. The callback and its state are used only here.
 * The result has metadata and tensor information, but cannot read tensor data.
 * The byte count is the complete source size, not the header size. */
typedef int (*aotx_modelfile_reader)(void *state, uint64_t offset, size_t bytes, void *out);
int aotx_modelfile_open_reader(const char *name, uint64_t bytes,
                              aotx_modelfile_reader reader, void *state,
                              aotx_modelfile **file);
uint64_t aotx_modelfile_file_bytes(const aotx_modelfile *file);
uint64_t aotx_modelfile_header_bytes(const aotx_modelfile *file);

/* Metadata by key. Each getter returns 0 when the key exists with a matching type. */
int aotx_modelfile_u32(const aotx_modelfile *file, const char *key, uint32_t *value);
int aotx_modelfile_u64(const aotx_modelfile *file, const char *key, uint64_t *value);
int aotx_modelfile_f32(const aotx_modelfile *file, const char *key, float *value);
int aotx_modelfile_string(const aotx_modelfile *file, const char *key, const char **value,
                          size_t *length);
int aotx_modelfile_strings(const aotx_modelfile *file, const char *key,
                           aotx_string_array *array);
int aotx_modelfile_i32s(const aotx_modelfile *file, const char *key, const int32_t **values,
                        uint64_t *count);
int aotx_modelfile_f32s(const aotx_modelfile *file, const char *key, const float **values,
                        uint64_t *count);
int aotx_modelfile_bools(const aotx_modelfile *file, const char *key, const uint8_t **values,
                         uint64_t *count);

/* The tensor table. */
uint64_t aotx_modelfile_tensor_count(const aotx_modelfile *file);
int aotx_modelfile_tensor(const aotx_modelfile *file, uint64_t index, aotx_tensor_info *info);
int aotx_modelfile_find(const aotx_modelfile *file, const char *name, aotx_tensor_info *info);
uint64_t aotx_modelfile_data_bytes(const aotx_modelfile *file);

/* Read tensor bytes into the caller's buffer. offset is from the start of the tensor bytes.
 * The caller streams a large tensor in pieces of its own size. Returns 0 on success. */
int aotx_modelfile_read(const aotx_modelfile *file, uint64_t offset, uint64_t bytes,
                        void *buffer);

/* The models manifest: one JSON line for each file, with identity and digest fields,
 * an optional wrap block, and a probe-layer fraction. A file whose sha256 differs
 * from its line is refused. */
#define AOTX_SHA256_HEX        65u

typedef struct aotx_manifest_entry {
    char name[64];
    char role[32];
    char path[256];
    char source[128];
    char revision[64];
    char license[32];
    uint64_t bytes;
    char sha256[AOTX_SHA256_HEX];
    aotx_wrap wrap;
    uint32_t wrap_present;
    uint32_t probe_numerator;
    uint32_t probe_denominator;
} aotx_manifest_entry;

/* Read the manifest beside the model files. Returns the entry count, or -1 on a bad line. */
int aotx_manifest_read(const char *dir, aotx_manifest_entry *entries, int max_entries);

/* Hash a file and compare it with its entry. Returns 0 when equal, 1 when different,
 * 2 when the file cannot be read. */
int aotx_manifest_digest(const char *text, unsigned char digest[32]);
int aotx_manifest_check(const char *dir, const aotx_manifest_entry *entry);
int aotx_modelfile_open_entry(const char *store, const aotx_manifest_entry *entry,
                              aotx_modelfile **file);

#ifdef __cplusplus
}
#endif

#endif
