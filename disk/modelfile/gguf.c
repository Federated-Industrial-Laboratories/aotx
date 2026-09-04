/* Purpose: Open a GGUF file, check its structure, and read its tensor bytes.
 * Owns: The head buffer of one open file, its tensor table, and its metadata pairs.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: From open to close. */
#include "disk/modelfile/gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The largest count of bytes that one read call asks for. A count above this is split,
 * because the kernel gives fewer bytes than a very large count asks for. */
#define AOTX_READ_PIECE (64u * 1024u * 1024u)

/* The first read makes a head of this size, and each later read doubles it. */
#define AOTX_HEAD_FIRST (64u * 1024u)

/* The read goes past the cursor by this count, so a run of small fields costs one read. */
#define AOTX_HEAD_AHEAD (1024u * 1024u)

int aotx_gguf_refuse(const aotx_modelfile *f, const char *reason)
{
    fprintf(stderr, "aotx_modelfile: %s: %s\n", f->path, reason);
    return 2;
}

/* Reads bytes from the file into the head. Returns 0, or 1 when the read fails. */
static int fill_head(aotx_modelfile *f, uint64_t end)
{
    while (f->filled < end) {
        uint64_t want = end - f->filled;
        ssize_t got;
        if (want > AOTX_READ_PIECE) {
            want = AOTX_READ_PIECE;
        }
        got = pread(f->fd, f->head + f->filled, (size_t)want, (off_t)f->filled);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "aotx_modelfile: %s: the file does not read\n", f->path);
            return 1;
        }
        if (got == 0) {
            fprintf(stderr, "aotx_modelfile: %s: the file is shorter than its size\n", f->path);
            return 1;
        }
        f->filled += (uint64_t)got;
    }
    return 0;
}

int aotx_gguf_need(aotx_modelfile *f, uint64_t bytes)
{
    uint64_t want;
    uint64_t end;
    /* The check is written this way so a length from the file cannot overflow the sum. */
    if (bytes > f->file_bytes || f->pos > f->file_bytes - bytes) {
        return aotx_gguf_refuse(f, "a field goes past the end of the file");
    }
    want = f->pos + bytes;
    if (want <= f->filled) {
        return 0;
    }
    if (want > AOTX_GGUF_HEAD_LIMIT) {
        return aotx_gguf_refuse(f, "the head of the file is too long");
    }
    if (want > f->head_bytes) {
        uint64_t size = (f->head_bytes > 0) ? f->head_bytes : AOTX_HEAD_FIRST;
        unsigned char *grown;
        /* The head limit above holds want at or below the limit. The doubling therefore
         * stops at or below the limit, and no clamp is necessary here. */
        while (size < want) {
            size *= 2u;
        }
        grown = (unsigned char *)realloc(f->head, (size_t)size);
        if (grown == NULL) {
            return aotx_gguf_refuse(f, "the memory for the head is not there");
        }
        f->head = grown;
        f->head_bytes = size;
    }
    end = want + AOTX_HEAD_AHEAD;
    if (end > f->head_bytes) {
        end = f->head_bytes;
    }
    if (end > f->file_bytes) {
        end = f->file_bytes;
    }
    /* The head must hold every byte that the cursor asked for, or a later read would take
     * bytes that no read filled. A file that gives fewer bytes than its size gives a
     * read fault here. */
    if (end < want) {
        end = want;
    }
    return fill_head(f, end);
}

int aotx_gguf_number(aotx_modelfile *f, unsigned width, uint64_t *out)
{
    uint64_t value = 0;
    unsigned i;
    int rc = aotx_gguf_need(f, width);
    if (rc != 0) {
        return rc;
    }
    /* The file holds the low byte first. The shift makes the value on a host of any byte
     * order, so the reader does not depend on the order of the host. */
    for (i = 0; i < width; i++) {
        value |= (uint64_t)f->head[f->pos + i] << (8u * i);
    }
    f->pos += width;
    *out = value;
    return 0;
}

int aotx_gguf_text(aotx_modelfile *f, const unsigned char **bytes, uint64_t *length)
{
    uint64_t len = 0;
    int rc = aotx_gguf_number(f, 8, &len);
    if (rc != 0) {
        return rc;
    }
    rc = aotx_gguf_need(f, len);
    if (rc != 0) {
        return rc;
    }
    *bytes = f->head + f->pos;
    *length = len;
    f->pos += len;
    return 0;
}

/* Gives the count of bytes of one tensor, from its dimensions and its type. A type that
 * this reader does not know gives zero bytes, because its layout is not known here. The
 * offset of every tensor comes from the file, so a zero here moves no other tensor. */
static int tensor_bytes(aotx_modelfile *f, aotx_tensor_info *t)
{
    uint64_t weights = 1;
    uint64_t block = 1;
    uint64_t per_block;
    uint32_t d;
    for (d = 0; d < t->dim_count; d++) {
        if (t->dims[d] == 0) {
            return aotx_gguf_refuse(f, "a tensor dimension is zero");
        }
        if (weights > UINT64_MAX / t->dims[d]) {
            return aotx_gguf_refuse(f, "the weight count of a tensor is too large");
        }
        weights *= t->dims[d];
    }
    switch (t->type) {
    case AOTX_TENSOR_F32:
        per_block = 4;
        break;
    case AOTX_TENSOR_F16:
        per_block = 2;
        break;
    case AOTX_TENSOR_Q4_0:
        block = AOTX_BLOCK_WEIGHTS;
        per_block = AOTX_Q4_0_BYTES;
        break;
    case AOTX_TENSOR_Q8_0:
        block = AOTX_BLOCK_WEIGHTS;
        per_block = AOTX_Q8_0_BYTES;
        break;
    case AOTX_TENSOR_Q4_K:
        block = AOTX_SUPER_WEIGHTS;
        per_block = AOTX_Q4_K_BYTES;
        break;
    case AOTX_TENSOR_Q5_K:
        block = AOTX_SUPER_WEIGHTS;
        per_block = AOTX_Q5_K_BYTES;
        break;
    case AOTX_TENSOR_Q6_K:
        block = AOTX_SUPER_WEIGHTS;
        per_block = AOTX_Q6_K_BYTES;
        break;
    default:
        t->bytes = 0;
        return 0;
    }
    /* The first dimension is the length of a row. A quantized row holds whole blocks, so a
     * row length that is not a multiple of the block has no layout. */
    if (block > 1 && (t->dims[0] % block) != 0) {
        return aotx_gguf_refuse(f, "a quantized row is not a whole count of blocks");
    }
    if ((weights / block) > UINT64_MAX / per_block) {
        return aotx_gguf_refuse(f, "the byte count of a tensor is too large");
    }
    t->bytes = (weights / block) * per_block;
    return 0;
}

/* Reads one tensor info: the name, the dimensions, the type, and the offset. */
static int tensor_info(aotx_modelfile *f, aotx_tensor_info *t)
{
    const unsigned char *name = NULL;
    uint64_t name_bytes = 0;
    uint64_t value = 0;
    uint32_t d;
    int rc = aotx_gguf_text(f, &name, &name_bytes);
    if (rc != 0) {
        return rc;
    }
    if (name_bytes >= AOTX_TENSOR_NAME_BYTES) {
        return aotx_gguf_refuse(f, "a tensor name is too long");
    }
    memcpy(t->name, name, (size_t)name_bytes);
    t->name[name_bytes] = '\0';
    rc = aotx_gguf_number(f, 4, &value);
    if (rc != 0) {
        return rc;
    }
    if (value == 0 || value > AOTX_TENSOR_DIMS) {
        return aotx_gguf_refuse(f, "a tensor has a dimension count that is not 1 to 4");
    }
    t->dim_count = (uint32_t)value;
    for (d = 0; d < t->dim_count; d++) {
        rc = aotx_gguf_number(f, 8, &t->dims[d]);
        if (rc != 0) {
            return rc;
        }
    }
    for (d = t->dim_count; d < AOTX_TENSOR_DIMS; d++) {
        t->dims[d] = 1;
    }
    rc = aotx_gguf_number(f, 4, &value);
    if (rc != 0) {
        return rc;
    }
    t->type = (uint32_t)value;
    rc = aotx_gguf_number(f, 8, &t->offset);
    if (rc != 0) {
        return rc;
    }
    return tensor_bytes(f, t);
}

/* Reads the tensor table, and then the alignment and the start of the tensor bytes. */
static int tensor_table(aotx_modelfile *f)
{
    uint64_t align = AOTX_GGUF_ALIGN;
    uint32_t small = 0;
    uint64_t pad;
    uint64_t i;
    int rc;
    if (f->tensor_count > 0) {
        f->tensors = (aotx_tensor_info *)calloc((size_t)f->tensor_count, sizeof(*f->tensors));
        if (f->tensors == NULL) {
            return aotx_gguf_refuse(f, "the memory for the tensor table is not there");
        }
    }
    for (i = 0; i < f->tensor_count; i++) {
        rc = tensor_info(f, &f->tensors[i]);
        if (rc != 0) {
            return rc;
        }
    }
    /* A file that gives the alignment key another type is malformed. The reader refuses
     * it, because a silent fall back to 32 would read a file that lies about its layout. */
    rc = aotx_modelfile_u32(f, "general.alignment", &small);
    if (rc == 0) {
        align = small;
    } else if (rc != 1) {
        return aotx_gguf_refuse(f, "the alignment key holds another type");
    }
    if (align == 0 || align > 65536u || (align & (align - 1u)) != 0) {
        return aotx_gguf_refuse(f, "the alignment is not a power of two from 1 to 65536");
    }
    f->alignment = (uint32_t)align;
    /* The cursor is never past the end, because every field read checks the end first. */
    pad = ((f->pos % align) == 0) ? 0 : (align - (f->pos % align));
    if (pad > f->file_bytes - f->pos) {
        return aotx_gguf_refuse(f, "the tensor bytes start past the end of the file");
    }
    f->data_offset = f->pos + pad;
    f->data_bytes = f->file_bytes - f->data_offset;
    for (i = 0; i < f->tensor_count; i++) {
        const aotx_tensor_info *t = &f->tensors[i];
        if ((t->offset % align) != 0) {
            return aotx_gguf_refuse(f, "a tensor offset is not on the alignment");
        }
        if (t->bytes > f->data_bytes || t->offset > f->data_bytes - t->bytes) {
            return aotx_gguf_refuse(f, "a tensor goes past the end of the tensor bytes");
        }
    }
    return 0;
}

/* Reads the file header: the magic, the version, and the two counts. */
static int file_header(aotx_modelfile *f)
{
    uint64_t value = 0;
    int rc = aotx_gguf_number(f, 4, &value);
    if (rc != 0) {
        return rc;
    }
    if (value != AOTX_GGUF_MAGIC) {
        return aotx_gguf_refuse(f, "the first four bytes are not GGUF");
    }
    rc = aotx_gguf_number(f, 4, &value);
    if (rc != 0) {
        return rc;
    }
    if (value != AOTX_GGUF_VERSION) {
        return aotx_gguf_refuse(f, "the version is not 3");
    }
    rc = aotx_gguf_number(f, 8, &f->tensor_count);
    if (rc != 0) {
        return rc;
    }
    rc = aotx_gguf_number(f, 8, &f->meta_count);
    if (rc != 0) {
        return rc;
    }
    /* A count that is larger than the file can hold is refused before any allocation. */
    if (f->tensor_count > f->file_bytes / AOTX_GGUF_TENSOR_LEAST) {
        return aotx_gguf_refuse(f, "the tensor count is larger than the file can hold");
    }
    if (f->meta_count > f->file_bytes / AOTX_GGUF_PAIR_LEAST) {
        return aotx_gguf_refuse(f, "the metadata count is larger than the file can hold");
    }
    return 0;
}

int aotx_modelfile_open(const char *path, aotx_modelfile **file)
{
    aotx_modelfile *f;
    struct stat st;
    int rc;
    if (path == NULL || file == NULL) {
        return 1;
    }
    *file = NULL;
    if (strlen(path) >= AOTX_GGUF_PATH_BYTES) {
        fprintf(stderr, "aotx_modelfile: the path is too long\n");
        return 1;
    }
    f = (aotx_modelfile *)calloc(1, sizeof(*f));
    if (f == NULL) {
        fprintf(stderr, "aotx_modelfile: %s: the memory for the file is not there\n", path);
        return 1;
    }
    memcpy(f->path, path, strlen(path) + 1);
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        fprintf(stderr, "aotx_modelfile: %s: the file does not open\n", path);
        free(f);
        return 1;
    }
    if (fstat(f->fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "aotx_modelfile: %s: the path is not a regular file\n", path);
        aotx_modelfile_close(f);
        return 1;
    }
    f->file_bytes = (uint64_t)st.st_size;
    rc = file_header(f);
    if (rc == 0) {
        rc = aotx_gguf_metadata(f);
    }
    if (rc == 0) {
        rc = tensor_table(f);
    }
    if (rc != 0) {
        aotx_modelfile_close(f);
        return rc;
    }
    /* Every value has a copy of its own, so the head is not needed after the parse. */
    free(f->head);
    f->head = NULL;
    f->head_bytes = 0;
    f->filled = 0;
    *file = f;
    return 0;
}

void aotx_modelfile_close(aotx_modelfile *file)
{
    if (file == NULL) {
        return;
    }
    aotx_gguf_metadata_release(file);
    free(file->tensors);
    free(file->head);
    if (file->fd >= 0) {
        close(file->fd);
    }
    free(file);
}

uint64_t aotx_modelfile_tensor_count(const aotx_modelfile *file)
{
    return (file != NULL) ? file->tensor_count : 0;
}

uint64_t aotx_modelfile_data_bytes(const aotx_modelfile *file)
{
    return (file != NULL) ? file->data_bytes : 0;
}

int aotx_modelfile_tensor(const aotx_modelfile *file, uint64_t index, aotx_tensor_info *info)
{
    if (file == NULL || info == NULL || index >= file->tensor_count) {
        return 1;
    }
    *info = file->tensors[index];
    return 0;
}

int aotx_modelfile_find(const aotx_modelfile *file, const char *name, aotx_tensor_info *info)
{
    uint64_t i;
    if (file == NULL || name == NULL || info == NULL) {
        return 1;
    }
    for (i = 0; i < file->tensor_count; i++) {
        if (strcmp(file->tensors[i].name, name) == 0) {
            *info = file->tensors[i];
            return 0;
        }
    }
    return 1;
}

int aotx_modelfile_read(const aotx_modelfile *file, uint64_t offset, uint64_t bytes, void *buffer)
{
    unsigned char *at = (unsigned char *)buffer;
    uint64_t left = bytes;
    uint64_t at_offset;
    if (file == NULL || (buffer == NULL && bytes > 0)) {
        return 1;
    }
    if (bytes > file->data_bytes || offset > file->data_bytes - bytes) {
        return aotx_gguf_refuse(file, "a read goes past the end of the tensor bytes");
    }
    at_offset = file->data_offset + offset;
    while (left > 0) {
        uint64_t want = (left < AOTX_READ_PIECE) ? left : AOTX_READ_PIECE;
        ssize_t got = pread(file->fd, at, (size_t)want, (off_t)at_offset);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "aotx_modelfile: %s: a tensor read failed\n", file->path);
            return 1;
        }
        if (got == 0) {
            fprintf(stderr, "aotx_modelfile: %s: a tensor read found the end\n", file->path);
            return 1;
        }
        at += (size_t)got;
        at_offset += (uint64_t)got;
        left -= (uint64_t)got;
    }
    return 0;
}
