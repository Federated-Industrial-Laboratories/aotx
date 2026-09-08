/* Purpose: Build model files with every value type and read them back through the reader.
 * Owns: The buffer of one built file, the plan of its tensors, and the counters of the run.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/modelfile/modelfile.h"

#include <fcntl.h>

#define AOTX_BUILD_BYTES (8u * 1024u * 1024u)

/* The head case claims a string longer than the head limit, in a file that is longer than
 * the string. The file is sparse, so only the first bytes hold data. */
#define AOTX_HEAD_CLAIM  280000000u
#define AOTX_HEAD_FILE   (320u * 1024u * 1024u)
#define AOTX_PAIRS       18
#define AOTX_PLAN_MAX    64

/* The defects that a built file can carry. Each one must give the code 2 at open. */
enum {
    D_NONE = 0, D_MAGIC, D_VERSION, D_META_COUNT, D_TENSOR_COUNT, D_KEY_LONG,
    D_STRING_PAST, D_META_TYPE, D_BOOL, D_ARRAY_IN_ARRAY, D_ARRAY_BIG, D_ALIGN,
    D_ALIGN_TYPE, D_DIM_COUNT, D_ROW_BLOCKS, D_NAME_LONG, D_OFFSET_ALIGN, D_TENSOR_PAST,
    D_TRUNCATE, D_LAST
};

static const char *defect_name[D_LAST] = {
    "none", "magic", "version", "metadata count", "tensor count", "long key",
    "string past the end", "value type", "boolean value", "array in an array",
    "array count", "alignment", "alignment type", "dimension count", "row blocks",
    "long tensor name", "tensor offset", "tensor past the end", "truncated file"
};

typedef struct build {
    unsigned char *at;
    size_t used;
    size_t size;
} build;

typedef struct plan {
    char name[AOTX_TENSOR_NAME_BYTES];
    uint32_t type;
    uint32_t dim_count;
    uint64_t dims[AOTX_TENSOR_DIMS];
    uint64_t offset;
    uint64_t bytes;
} plan;

static void raw(build *b, const void *data, size_t bytes)
{
    if (b->used + bytes > b->size) {
        printf("FAIL the build buffer is too small\n");
        exit(1);
    }
    memcpy(b->at + b->used, data, bytes);
    b->used += bytes;
}

/* Writes a whole number with the low byte first, which is the order that the format uses. */
static void num(build *b, uint64_t value, unsigned width)
{
    unsigned char bytes[8];
    unsigned i;
    for (i = 0; i < width; i++) {
        bytes[i] = (unsigned char)(value >> (8u * i));
    }
    raw(b, bytes, width);
}

static void text(build *b, const char *value)
{
    num(b, strlen(value), 8);
    raw(b, value, strlen(value));
}

static void real32(build *b, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    num(b, bits, 4);
}

static void real64(build *b, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    num(b, bits, 8);
}

/* Gives the count of bytes of a tensor, from the same rule that the reader applies. */
static uint64_t plan_bytes(uint32_t type, const uint64_t *dims, uint32_t dim_count)
{
    uint64_t weights = 1;
    uint32_t d;
    for (d = 0; d < dim_count; d++) {
        weights *= dims[d];
    }
    if (type == AOTX_TENSOR_F32) {
        return weights * 4u;
    }
    if (type == AOTX_TENSOR_F16) {
        return weights * 2u;
    }
    if (type == AOTX_TENSOR_Q4_0) {
        return (weights / AOTX_BLOCK_WEIGHTS) * AOTX_Q4_0_BYTES;
    }
    if (type == AOTX_TENSOR_Q4_1) {
        return (weights / 32u) * 20u;
    }
    if (type == AOTX_TENSOR_Q5_0) {
        return (weights / 32u) * 22u;
    }
    if (type == AOTX_TENSOR_Q5_1) {
        return (weights / 32u) * 24u;
    }
    if (type == AOTX_TENSOR_Q2_K) {
        return (weights / 256u) * 84u;
    }
    if (type == AOTX_TENSOR_Q3_K) {
        return (weights / 256u) * 110u;
    }
    return (weights / AOTX_BLOCK_WEIGHTS) * AOTX_Q8_0_BYTES;
}

/* Lays out the tensors. Every tensor has a different shape and a different byte count, so
 * a wrong index or a wrong offset cannot hide. */
static uint64_t make_plan(plan *p, int count, int defect)
{
    static const uint32_t types[] = {
        AOTX_TENSOR_F32, AOTX_TENSOR_F16, AOTX_TENSOR_Q4_0, AOTX_TENSOR_Q8_0,
        AOTX_TENSOR_Q4_1, AOTX_TENSOR_Q5_0, AOTX_TENSOR_Q5_1,
        AOTX_TENSOR_Q2_K, AOTX_TENSOR_Q3_K
    };
    uint64_t at = 0;
    int i;
    uint32_t d;
    for (i = 0; i < count; i++) {
        p[i].type = types[i % (sizeof types / sizeof types[0])];
        p[i].dim_count = 1u + (uint32_t)(i % 3);
        p[i].dims[0] = 32u * (uint64_t)(i + 1);
        if (p[i].type == AOTX_TENSOR_Q2_K || p[i].type == AOTX_TENSOR_Q3_K) {
            p[i].dims[0] = 256u * (uint64_t)(i + 1);
        }
        for (d = 1; d < AOTX_TENSOR_DIMS; d++) {
            p[i].dims[d] = (d < p[i].dim_count) ? (uint64_t)((i + (int)d) % 3 + 1) : 1u;
        }
        if (defect == D_ROW_BLOCKS && p[i].type == AOTX_TENSOR_Q8_0) {
            p[i].dims[0] += 1u;
        }
        snprintf(p[i].name, sizeof(p[i].name), "tensor.%d.weight", i);
        p[i].bytes = plan_bytes(p[i].type, p[i].dims, p[i].dim_count);
        p[i].offset = at;
        at += (p[i].bytes + 31u) & ~(uint64_t)31u;
    }
    if (defect == D_OFFSET_ALIGN && count > 1) {
        /* The tensor bytes grow by one place, so this defect moves the offset off the
         * alignment without also moving the tensor past the end. */
        p[1].offset += 1u;
        at += 32u;
    }
    if (defect == D_TENSOR_PAST && count > 0) {
        p[count - 1].offset += at;
    }
    /* The count of tensor bytes comes from the plan before a defect moves an offset.
     * A tensor that a defect moves therefore goes past the end of the tensor bytes. */
    return at;
}

/* The byte at place k of tensor i. Every tensor holds different content. */
static unsigned char tensor_byte(int i, uint64_t k)
{
    return (unsigned char)(((uint64_t)(i + 1) * 131u + k * 17u + (k >> 8)) & 0xffu);
}

static void write_pairs(build *b, int defect, int elements)
{
    char key[512];
    char value[64];
    int i;
    text(b, "general.architecture");
    num(b, AOTX_GGUF_STRING, 4);
    if (defect == D_STRING_PAST) {
        num(b, (uint64_t)1 << 40, 8);
        raw(b, "aotx.test", 9);
    } else {
        text(b, "aotx.test");
    }
    text(b, "general.alignment");
    if (defect == D_ALIGN_TYPE) {
        /* The key holds a string, which is not the type that the format gives it. */
        num(b, AOTX_GGUF_STRING, 4);
        text(b, "32");
    } else {
        num(b, AOTX_GGUF_U32, 4);
        num(b, (defect == D_ALIGN) ? 33u : AOTX_GGUF_ALIGN, 4);
    }
    text(b, "tokenizer.ggml.pre");
    num(b, AOTX_GGUF_STRING, 4);
    text(b, "qwen2");
    if (defect == D_KEY_LONG) {
        memset(key, 'k', sizeof(key) - 1);
        key[sizeof(key) - 1] = '\0';
        text(b, key);
    } else {
        text(b, "test.u8");
    }
    num(b, (defect == D_META_TYPE) ? 99u : AOTX_GGUF_U8, 4);
    num(b, 200u, 1);
    text(b, "test.i8");
    num(b, AOTX_GGUF_I8, 4);
    num(b, (uint64_t)(uint8_t)(-100), 1);
    text(b, "test.u16");
    num(b, AOTX_GGUF_U16, 4);
    num(b, 60000u, 2);
    text(b, "test.i16");
    num(b, AOTX_GGUF_I16, 4);
    num(b, (uint64_t)(uint16_t)(-30000), 2);
    text(b, "test.u32");
    num(b, AOTX_GGUF_U32, 4);
    num(b, 4000000000u, 4);
    text(b, "test.i32");
    num(b, AOTX_GGUF_I32, 4);
    num(b, (uint64_t)(uint32_t)(-2000000000), 4);
    text(b, "test.f32");
    num(b, AOTX_GGUF_F32, 4);
    real32(b, 1.5f);
    text(b, "test.bool");
    num(b, AOTX_GGUF_BOOL, 4);
    num(b, (defect == D_BOOL) ? 7u : 1u, 1);
    text(b, "test.u64");
    num(b, AOTX_GGUF_U64, 4);
    num(b, 18000000000000000000ull, 8);
    text(b, "test.i64");
    num(b, AOTX_GGUF_I64, 4);
    num(b, (uint64_t)(-9000000000000000000ll), 8);
    text(b, "test.f64");
    num(b, AOTX_GGUF_F64, 4);
    real64(b, 2.25);
    text(b, "test.strings");
    num(b, AOTX_GGUF_ARRAY, 4);
    num(b, (defect == D_ARRAY_IN_ARRAY) ? AOTX_GGUF_ARRAY : AOTX_GGUF_STRING, 4);
    num(b, (uint64_t)elements, 8);
    for (i = 0; i < elements; i++) {
        snprintf(value, sizeof(value), "token.%d", i);
        text(b, value);
    }
    text(b, "test.i32s");
    num(b, AOTX_GGUF_ARRAY, 4);
    num(b, AOTX_GGUF_I32, 4);
    num(b, (defect == D_ARRAY_BIG) ? 100000000ull : (uint64_t)elements, 8);
    for (i = 0; i < elements; i++) {
        num(b, (uint64_t)(uint32_t)(i - elements / 2), 4);
    }
    text(b, "test.f32s");
    num(b, AOTX_GGUF_ARRAY, 4);
    num(b, AOTX_GGUF_F32, 4);
    num(b, (uint64_t)elements, 8);
    for (i = 0; i < elements; i++) {
        real32(b, (float)i * 0.5f + 0.25f);
    }
    text(b, "test.u8s");
    num(b, AOTX_GGUF_ARRAY, 4);
    num(b, AOTX_GGUF_U8, 4);
    num(b, (uint64_t)elements, 8);
    for (i = 0; i < elements; i++) {
        num(b, (uint64_t)(unsigned)(i & 0xff), 1);
    }
}

/* Builds one file into the buffer. Gives the count of bytes, and the place of the tensor
 * bytes in head_end. */
static size_t make_file(build *b, int defect, int tensors, int elements, plan *p,
                        uint64_t *head_end)
{
    uint64_t data_offset;
    uint64_t data_bytes;
    int i;
    uint32_t d;
    b->used = 0;
    data_bytes = make_plan(p, tensors, defect);
    raw(b, (defect == D_MAGIC) ? "GGUX" : "GGUF", 4);
    num(b, (defect == D_VERSION) ? 2u : AOTX_GGUF_VERSION, 4);
    num(b, (defect == D_TENSOR_COUNT) ? ((uint64_t)1 << 40) : (uint64_t)tensors, 8);
    num(b, (defect == D_META_COUNT) ? ((uint64_t)1 << 40) : AOTX_PAIRS, 8);
    write_pairs(b, defect, elements);
    for (i = 0; i < tensors; i++) {
        if (defect == D_NAME_LONG && i == 0) {
            char long_name[200];
            memset(long_name, 'n', sizeof(long_name) - 1);
            long_name[sizeof(long_name) - 1] = '\0';
            text(b, long_name);
        } else {
            text(b, p[i].name);
        }
        num(b, (defect == D_DIM_COUNT && i == 0) ? 0u : p[i].dim_count, 4);
        for (d = 0; d < p[i].dim_count; d++) {
            num(b, p[i].dims[d], 8);
        }
        num(b, p[i].type, 4);
        num(b, p[i].offset, 8);
    }
    data_offset = (b->used + 31u) & ~(size_t)31u;
    while (b->used < data_offset) {
        num(b, 0, 1);
    }
    *head_end = data_offset;
    for (i = 0; i < tensors; i++) {
        uint64_t k;
        uint64_t at = data_offset + p[i].offset;
        for (k = 0; k < p[i].bytes && at + k < b->size; k++) {
            b->at[at + k] = tensor_byte(i, k);
        }
    }
    b->used = (size_t)(data_offset + data_bytes);
    return b->used;
}

static int put_file(const char *path, const unsigned char *bytes, size_t count)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return -1;
    }
    if (fwrite(bytes, 1, count, f) != count) {
        fclose(f);
        return -1;
    }
    return (fclose(f) == 0) ? 0 : -1;
}

/* Reads back one good file and checks every value against the plan. */
static void check_file(const char *path, int tensors, int elements, const plan *p)
{
    aotx_modelfile *file = NULL;
    aotx_string_array strings;
    aotx_tensor_info info;
    const int32_t *whole = NULL;
    const float *reals = NULL;
    const char *word = NULL;
    unsigned char *buffer;
    uint64_t count = 0;
    uint32_t u32 = 0;
    uint64_t u64 = 0;
    float f32 = 0.0f;
    size_t length = 0;
    int i;
    CHECK(aotx_modelfile_open(path, &file) == 0, "the file does not open");
    if (file == NULL) {
        return;
    }
    CHECK(aotx_modelfile_string(file, "general.architecture", &word, &length) == 0 &&
          strcmp(word, "aotx.test") == 0, "the architecture is wrong");
    CHECK(length == 9, "the architecture length is %d", (int)length);
    CHECK(aotx_modelfile_string(file, "tokenizer.ggml.pre", &word, &length) == 0
          && length == 5 && strcmp(word, "qwen2") == 0, "the pre-tokenizer value is wrong");
    CHECK(aotx_modelfile_u32(file, "test.u8", &u32) == 0 && u32 == 200, "u8 gives %u", u32);
    CHECK(aotx_modelfile_u32(file, "test.u16", &u32) == 0 && u32 == 60000, "u16 gives %u", u32);
    CHECK(aotx_modelfile_u32(file, "test.u32", &u32) == 0 && u32 == 4000000000u,
          "u32 gives %u", u32);
    CHECK(aotx_modelfile_u64(file, "test.u64", &u64) == 0 && u64 == 18000000000000000000ull,
          "u64 gives %llu", (unsigned long long)u64);
    CHECK(aotx_modelfile_u32(file, "test.bool", &u32) == 0 && u32 == 1, "bool gives %u", u32);
    CHECK(aotx_modelfile_f32(file, "test.f32", &f32) == 0 && f32 == 1.5f, "f32 gives %f",
          (double)f32);
    CHECK(aotx_modelfile_f32(file, "test.f64", &f32) == 0 && f32 == 2.25f, "f64 gives %f",
          (double)f32);
    /* A value with a sign that is below zero has no value without a sign. */
    CHECK(aotx_modelfile_u32(file, "test.i8", &u32) == 2, "a negative value gives a number");
    CHECK(aotx_modelfile_u64(file, "test.i64", &u64) == 2, "a negative value gives a number");
    CHECK(aotx_modelfile_u32(file, "test.absent", &u32) == 1, "an absent key gives a value");
    CHECK(aotx_modelfile_string(file, "test.u8", &word, &length) == 2,
          "a number gives a string");
    CHECK(aotx_modelfile_strings(file, "test.strings", &strings) == 0, "the strings are absent");
    CHECK(strings.count == (uint64_t)elements, "the string count is %llu",
          (unsigned long long)strings.count);
    for (i = 0; i < elements && strings.count == (uint64_t)elements; i++) {
        char want[64];
        uint64_t from = strings.offsets[i];
        uint64_t to = strings.offsets[i + 1];
        snprintf(want, sizeof(want), "token.%d", i);
        CHECK(to - from == strlen(want) &&
              memcmp(strings.bytes + from, want, (size_t)(to - from)) == 0,
              "string %d is not %s", i, want);
    }
    CHECK(aotx_modelfile_i32s(file, "test.i32s", &whole, &count) == 0 &&
          count == (uint64_t)elements, "the whole number array is absent");
    for (i = 0; i < elements && whole != NULL && count == (uint64_t)elements; i++) {
        CHECK(whole[i] == i - elements / 2, "element %d of the array is %d", i, whole[i]);
    }
    CHECK(aotx_modelfile_f32s(file, "test.f32s", &reals, &count) == 0 &&
          count == (uint64_t)elements, "the real number array is absent");
    for (i = 0; i < elements && reals != NULL && count == (uint64_t)elements; i++) {
        CHECK(reals[i] == (float)i * 0.5f + 0.25f, "element %d of the array is %f", i,
              (double)reals[i]);
    }
    /* An array of a type that no getter gives is read past, and the getter refuses it. */
    CHECK(aotx_modelfile_i32s(file, "test.u8s", &whole, &count) == 2,
          "an array of bytes gives whole numbers");
    CHECK(aotx_modelfile_tensor_count(file) == (uint64_t)tensors, "the tensor count is %llu",
          (unsigned long long)aotx_modelfile_tensor_count(file));
    buffer = (unsigned char *)malloc(1024u * 1024u);
    CHECK(buffer != NULL, "the read buffer is not there");
    for (i = 0; i < tensors && buffer != NULL; i++) {
        uint64_t k;
        int same = 1;
        CHECK(aotx_modelfile_tensor(file, (uint64_t)i, &info) == 0, "tensor %d is absent", i);
        CHECK(strcmp(info.name, p[i].name) == 0, "tensor %d has the name %s", i, info.name);
        CHECK(info.type == p[i].type && info.dim_count == p[i].dim_count &&
              info.offset == p[i].offset && info.bytes == p[i].bytes,
              "tensor %d has another shape", i);
        CHECK(aotx_modelfile_find(file, p[i].name, &info) == 0, "tensor %s is not found",
              p[i].name);
        CHECK(aotx_modelfile_read(file, info.offset, info.bytes, buffer) == 0,
              "tensor %d does not read", i);
        for (k = 0; k < info.bytes; k++) {
            if (buffer[k] != tensor_byte(i, k)) {
                same = 0;
            }
        }
        CHECK(same == 1, "tensor %d holds other bytes", i);
        /* The caller streams a tensor in pieces of its own size. */
        if (info.bytes >= 3) {
            uint64_t piece = info.bytes / 3;
            CHECK(aotx_modelfile_read(file, info.offset, piece, buffer) == 0 &&
                  aotx_modelfile_read(file, info.offset + piece, piece, buffer + piece) == 0,
                  "tensor %d does not read in pieces", i);
            same = 1;
            for (k = 0; k < piece * 2u; k++) {
                if (buffer[k] != tensor_byte(i, k)) {
                    same = 0;
                }
            }
            CHECK(same == 1, "tensor %d gives other bytes in pieces", i);
        }
    }
    CHECK(aotx_modelfile_find(file, "tensor.absent", &info) == 1, "an absent tensor is found");
    CHECK(aotx_modelfile_tensor(file, (uint64_t)tensors, &info) == 1,
          "a tensor past the table is there");
    /* A read that the tensor bytes cannot hold is refused with the code of a bad
     * request. That code is not the code of a disk that does not read. */
    CHECK(aotx_modelfile_read(file, aotx_modelfile_data_bytes(file), 1, buffer) == 2,
          "a read past the tensor bytes is accepted");
    CHECK(aotx_modelfile_read(file, 0, aotx_modelfile_data_bytes(file) + 1u, buffer) == 2,
          "a long read past the tensor bytes is accepted");
    free(buffer);
    aotx_modelfile_close(file);
}

static void round_trip(const char *dir, int tensors, int elements, build *b, plan *p)
{
    char path[512];
    uint64_t head_end = 0;
    size_t bytes;
    snprintf(path, sizeof(path), "%s/good-%d.gguf", dir, tensors);
    bytes = make_file(b, D_NONE, tensors, elements, p, &head_end);
    CHECK(put_file(path, b->at, bytes) == 0, "the file does not write");
    printf("gguf_test: built %s, %llu bytes, %d tensors, %d array elements\n", path,
           (unsigned long long)bytes, tensors, elements);
    check_file(path, tensors, elements, p);
}

/* Every defect must give the code 2, and no defect may end the program. */
static void guards(const char *dir, build *b, plan *p)
{
    char path[512];
    uint64_t head_end = 0;
    int defect;
    for (defect = D_MAGIC; defect < D_LAST; defect++) {
        aotx_modelfile *file = NULL;
        /* The alignment case holds one tensor, whose offset is zero. A file with more
         * tensors would meet the offset guard first, and the alignment guard would not
         * be the guard that the case examines. */
        int tensors = (defect == D_ALIGN) ? 1 : 8;
        size_t bytes = make_file(b, defect, tensors, 8, p, &head_end);
        int rc;
        if (defect == D_TRUNCATE) {
            bytes = (size_t)head_end / 2u;
        }
        snprintf(path, sizeof(path), "%s/defect-%d.gguf", dir, defect);
        CHECK(put_file(path, b->at, bytes) == 0, "the file of defect %d does not write", defect);
        rc = aotx_modelfile_open(path, &file);
        CHECK(rc == 2, "the defect %s gives the code %d", defect_name[defect], rc);
        CHECK(file == NULL, "the defect %s gives an open file", defect_name[defect]);
        aotx_modelfile_close(file);
    }
}

/* A hostile file must give a code, and must not end the program. A corruption inside a
 * string or inside the tensor bytes leaves a file that the format accepts. The case
 * therefore counts the codes and walks every file that opens. */
static void fuzz(const char *dir, build *b, plan *p)
{
    char path[512];
    unsigned char *copy;
    uint64_t head_end = 0;
    size_t bytes = make_file(b, D_NONE, 8, 8, p, &head_end);
    unsigned seed = 20260827u;
    int opened = 0;
    int refused = 0;
    int faulted = 0;
    int i;
    copy = (unsigned char *)malloc(bytes);
    CHECK(copy != NULL, "the copy buffer is not there");
    if (copy == NULL) {
        return;
    }
    snprintf(path, sizeof(path), "%s/fuzz.gguf", dir);
    for (i = 0; i < 64; i++) {
        aotx_modelfile *file = NULL;
        size_t count = bytes;
        int hits;
        int rc;
        memcpy(copy, b->at, bytes);
        seed = seed * 1103515245u + 12345u;
        hits = 1 + (int)((seed >> 16) % 4u);
        while (hits-- > 0) {
            size_t place;
            seed = seed * 1103515245u + 12345u;
            place = (size_t)((seed >> 8) % head_end);
            seed = seed * 1103515245u + 12345u;
            copy[place] = (unsigned char)(seed >> 16);
        }
        if ((i % 8) == 7) {
            seed = seed * 1103515245u + 12345u;
            count = (size_t)((seed >> 8) % bytes);
        }
        CHECK(put_file(path, copy, count) == 0, "the fuzz file does not write");
        rc = aotx_modelfile_open(path, &file);
        if (rc == 0) {
            /* The file opens, so every getter and every tensor read must hold. */
            uint64_t n = aotx_modelfile_tensor_count(file);
            uint64_t t;
            aotx_string_array strings;
            const char *word = NULL;
            unsigned char small[4096];
            size_t length = 0;
            opened++;
            (void)aotx_modelfile_string(file, "general.architecture", &word, &length);
            strings.count = 0;
            if (aotx_modelfile_strings(file, "test.strings", &strings) == 0) {
                uint64_t s_index;
                unsigned sum = 0;
                for (s_index = 0; s_index < strings.count; s_index++) {
                    uint64_t from = strings.offsets[s_index];
                    uint64_t to = strings.offsets[s_index + 1];
                    while (from < to) {
                        sum += strings.bytes[from++];
                    }
                }
                (void)sum;
            }
            for (t = 0; t < n; t++) {
                aotx_tensor_info info;
                if (aotx_modelfile_tensor(file, t, &info) == 0) {
                    uint64_t want = (info.bytes < sizeof(small)) ? info.bytes : sizeof(small);
                    (void)aotx_modelfile_read(file, info.offset, want, small);
                }
            }
            aotx_modelfile_close(file);
        } else if (rc == 2) {
            refused++;
        } else {
            faulted++;
        }
        CHECK(rc == 0 || rc == 1 || rc == 2, "the fuzz case %d gives the code %d", i, rc);
        CHECK(rc == 0 || file == NULL, "the fuzz case %d gives an open file", i);
    }
    printf("gguf_test: fuzz cases 64, refused %d, opened %d, read fault %d\n", refused,
           opened, faulted);
    CHECK(refused >= 8, "the fuzz refused only %d of 64 cases", refused);
    free(copy);
}

/* The head of a file must not go into memory when it is longer than the limit. The file is
 * made sparse, so a file of 320 MB costs almost no disk. */
static void head_limit(const char *dir)
{
    char path[512];
    unsigned char header[64];
    build small;
    aotx_modelfile *file = NULL;
    ssize_t put;
    int fd;
    int rc;
    snprintf(path, sizeof(path), "%s/head.gguf", dir);
    small.at = header;
    small.size = sizeof(header);
    small.used = 0;
    raw(&small, "GGUF", 4);
    num(&small, AOTX_GGUF_VERSION, 4);
    num(&small, 0, 8);
    num(&small, 1, 8);
    text(&small, "big");
    num(&small, AOTX_GGUF_STRING, 4);
    num(&small, AOTX_HEAD_CLAIM, 8);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0, "the head file does not open");
    if (fd < 0) {
        return;
    }
    put = write(fd, header, small.used);
    CHECK(put == (ssize_t)small.used, "the head of the head file does not write");
    CHECK(ftruncate(fd, (off_t)AOTX_HEAD_FILE) == 0, "the head file does not grow");
    close(fd);
    rc = aotx_modelfile_open(path, &file);
    CHECK(rc == 2, "a head longer than the limit gives the code %d", rc);
    CHECK(file == NULL, "a head longer than the limit gives an open file");
    aotx_modelfile_close(file);
    remove(path);
}

int main(void)
{
    char dir[256];
    build b;
    plan *p;
    b.at = (unsigned char *)malloc(AOTX_BUILD_BYTES);
    b.size = AOTX_BUILD_BYTES;
    b.used = 0;
    p = (plan *)calloc(AOTX_PLAN_MAX, sizeof(plan));
    if (b.at == NULL || p == NULL || aotx_temp_dir(dir, sizeof(dir)) != 0) {
        printf("FAIL the test cannot start\n");
        return 1;
    }
    memset(b.at, 0, AOTX_BUILD_BYTES);
    round_trip(dir, 1, 1, &b, p);
    memset(b.at, 0, AOTX_BUILD_BYTES);
    round_trip(dir, 64, 64, &b, p);
    guards(dir, &b, p);
    head_limit(dir);
    fuzz(dir, &b, p);
    aotx_remove_tree(dir);
    free(b.at);
    free(p);
    return aotx_report("gguf_test", 500);
}
