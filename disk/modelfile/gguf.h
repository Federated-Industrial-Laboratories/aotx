/* Purpose: Declare the parts that the two model file translation units share.
 * Owns: Nothing; the open file structure holds every allocation that these parts make.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: From open to close. */
#ifndef AOTX_MODELFILE_GGUF_H
#define AOTX_MODELFILE_GGUF_H

#include "disk/modelfile/modelfile.h"

/* The head is the part of the file before the tensor bytes: the file header, the metadata
 * pairs, and the tensor table. A file whose head is longer than this limit is refused,
 * because the head goes into memory in one piece. */
#define AOTX_GGUF_HEAD_LIMIT   (256u * 1024u * 1024u)

/* All parser allocations together must fit this bound. */
#define AOTX_GGUF_MEMORY_LIMIT (512u * 1024u * 1024u)

/* The least bytes that one metadata pair and one tensor info can occupy. The counts in the
 * file header are checked against these, so a large count cannot make a large allocation. */
#define AOTX_GGUF_PAIR_LEAST   12u
#define AOTX_GGUF_TENSOR_LEAST 24u

#define AOTX_GGUF_KEY_BYTES    256u
#define AOTX_GGUF_PATH_BYTES   512u

/* One metadata pair. The value is a scalar, a string, or an array. The reader copies every
 * value at open, so no pointer here points into the head buffer. */
typedef struct aotx_meta {
    char *key;                  /* the key, with an end byte */
    uint32_t type;              /* the type of the value */
    uint32_t element_type;      /* the type of one element, when the value is an array */
    uint64_t count;             /* the count of elements, when the value is an array */
    uint64_t u;                 /* a whole number value, with no sign */
    int64_t i;                  /* a whole number value, with a sign */
    double f;                   /* a real number value */
    char *text;                 /* a string value, with an end byte */
    uint64_t text_bytes;        /* the length of the string value, without the end byte */
    uint8_t *run;               /* the strings of an array, one after the other */
    uint64_t *offsets;          /* count + 1 entries into the run */
    int32_t *i32;               /* the elements of an array of whole numbers */
    float *f32;                 /* the elements of an array of real numbers */
} aotx_meta;

struct aotx_modelfile {
    int fd;
    uint64_t file_bytes;
    uint64_t data_offset;       /* the first byte of the tensor bytes */
    uint64_t data_bytes;        /* the count of tensor bytes */
    uint32_t alignment;         /* from general.alignment, or 32 */
    uint64_t meta_count;
    aotx_meta *meta;
    uint64_t tensor_count;
    aotx_tensor_info *tensors;
    unsigned char *head;        /* the file bytes from zero to filled, released after open */
    uint64_t head_bytes;        /* the size of the head allocation */
    uint64_t filled;            /* the count of file bytes that the head holds */
    uint64_t pos;               /* the cursor of the parse */
    uint64_t allocated;
    aotx_modelfile_reader reader;
    void *reader_state;
    char path[AOTX_GGUF_PATH_BYTES];
};

/* Prints the cause on the error output and gives 2, which is the code of a bad format. */
int aotx_gguf_refuse(const aotx_modelfile *f, const char *reason);

/* Makes the head hold the bytes from the cursor. Returns 0, or 2 when the file is too
 * short, the head limit is reached, or the memory is not there. */
int aotx_gguf_need(aotx_modelfile *f, uint64_t bytes);

/* Charge an allocation before it is made. Released values keep their charge until close. */
int aotx_gguf_charge(aotx_modelfile *f, uint64_t bytes);

/* Reads a little-endian whole number of one, two, four, or eight bytes and moves the
 * cursor. The byte order of the host does not change the result. Returns 0 or 2. */
int aotx_gguf_number(aotx_modelfile *f, unsigned width, uint64_t *out);

/* Reads a string: a length of eight bytes and the bytes of the string. The bytes stay in
 * the head, and the caller copies what it keeps. Returns 0 or 2. */
int aotx_gguf_text(aotx_modelfile *f, const unsigned char **bytes, uint64_t *length);

/* Reads the metadata pairs into the file structure. Returns 0 or 2. */
int aotx_gguf_metadata(aotx_modelfile *f);

/* Releases every allocation that the metadata pairs hold. */
void aotx_gguf_metadata_release(aotx_modelfile *f);

#endif
