/* Purpose: Read the metadata pairs of a GGUF file and give their values by key.
 * Owns: The key, the string, the array run, and the element copies of each pair.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: From open to close. */
#include "disk/modelfile/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The largest count of elements that one array can hold. The token array of a model of this
 * class holds about 152,000 strings, so this limit is far above what a model file needs. */
#define AOTX_GGUF_ARRAY_LIMIT (8u * 1024u * 1024u)

/* The first size of the byte run of a string array, which doubles as the run grows. */
#define AOTX_GGUF_RUN_FIRST   4096u

/* The getters give these codes. The header states that 0 means success. */
#define AOTX_META_MISSING     1
#define AOTX_META_TYPE        2

/* Gives the count of bytes of one value type, or zero when the type is not a fixed width. */
static unsigned width_of(uint32_t type)
{
    switch (type) {
    case AOTX_GGUF_U8:
    case AOTX_GGUF_I8:
    case AOTX_GGUF_BOOL:
        return 1;
    case AOTX_GGUF_U16:
    case AOTX_GGUF_I16:
        return 2;
    case AOTX_GGUF_U32:
    case AOTX_GGUF_I32:
    case AOTX_GGUF_F32:
        return 4;
    case AOTX_GGUF_U64:
    case AOTX_GGUF_I64:
    case AOTX_GGUF_F64:
        return 8;
    default:
        return 0;
    }
}

static int is_signed(uint32_t type)
{
    return (type == AOTX_GGUF_I8 || type == AOTX_GGUF_I16 || type == AOTX_GGUF_I32 ||
            type == AOTX_GGUF_I64);
}

/* Makes the value of a whole number that has a sign. The host holds a negative whole
 * number in two's complement form, which is the form that the file also holds. */
static int64_t signed_value(uint64_t raw, unsigned width)
{
    uint64_t sign;
    if (width >= 8) {
        return (int64_t)raw;
    }
    sign = (uint64_t)1 << (width * 8u - 1u);
    if ((raw & sign) != 0) {
        raw |= ~(((uint64_t)1 << (width * 8u)) - 1u);
    }
    return (int64_t)raw;
}

/* The file holds a real number in the form of IEEE 754, with the low byte first. */
static float float_of(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double double_of(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Reads one scalar value into the pair. Returns 0 or 2. */
static int scalar(aotx_modelfile *f, aotx_meta *m, uint32_t type)
{
    unsigned width = width_of(type);
    uint64_t raw = 0;
    int rc;
    if (width == 0) {
        return aotx_gguf_refuse(f, "a metadata value has a type that is not known");
    }
    rc = aotx_gguf_number(f, width, &raw);
    if (rc != 0) {
        return rc;
    }
    m->u = raw;
    m->i = is_signed(type) ? signed_value(raw, width) : (int64_t)raw;
    if (type == AOTX_GGUF_F32) {
        m->f = (double)float_of((uint32_t)raw);
    } else if (type == AOTX_GGUF_F64) {
        m->f = double_of(raw);
    } else {
        m->f = 0.0;
    }
    if (type == AOTX_GGUF_BOOL && raw > 1u) {
        return aotx_gguf_refuse(f, "a boolean value is not 0 or 1");
    }
    return 0;
}

/* Reads a string value and keeps a copy with an end byte. Returns 0 or 2. */
static int string_value(aotx_modelfile *f, aotx_meta *m)
{
    const unsigned char *bytes = NULL;
    uint64_t length = 0;
    int rc = aotx_gguf_text(f, &bytes, &length);
    if (rc != 0) {
        return rc;
    }
    if (aotx_gguf_charge(f, length + 1u) != 0) return 2;
    m->text = (char *)malloc((size_t)length + 1u);
    if (m->text == NULL) {
        return aotx_gguf_refuse(f, "the memory for a string value is not there");
    }
    memcpy(m->text, bytes, (size_t)length);
    m->text[length] = '\0';
    m->text_bytes = length;
    return 0;
}

/* Reads an array of strings into one byte run and one table of offsets. Returns 0 or 2. */
static int string_array(aotx_modelfile *f, aotx_meta *m)
{
    uint64_t run_bytes = 0;
    uint64_t used = 0;
    uint64_t i;
    if (m->count > (f->file_bytes - f->pos) / 8u ||
        m->count > (AOTX_GGUF_HEAD_LIMIT - f->pos) / 8u)
        return aotx_gguf_refuse(f, "a string array count exceeds the remaining header");
    if (aotx_gguf_charge(f, (m->count + 1u) * sizeof(uint64_t)) != 0) return 2;
    m->offsets = (uint64_t *)calloc((size_t)m->count + 1u, sizeof(uint64_t));
    if (m->offsets == NULL) {
        return aotx_gguf_refuse(f, "the memory for a string array is not there");
    }
    for (i = 0; i < m->count; i++) {
        const unsigned char *bytes = NULL;
        uint64_t length = 0;
        int rc = aotx_gguf_text(f, &bytes, &length);
        if (rc != 0) {
            return rc;
        }
        if (used + length > run_bytes) {
            uint64_t size = (run_bytes > 0) ? run_bytes : AOTX_GGUF_RUN_FIRST;
            uint8_t *grown;
            while (size < used + length) {
                size *= 2u;
            }
            if (aotx_gguf_charge(f, size - run_bytes) != 0) return 2;
            grown = (uint8_t *)realloc(m->run, (size_t)size);
            if (grown == NULL) {
                return aotx_gguf_refuse(f, "the memory for a string array is not there");
            }
            m->run = grown;
            run_bytes = size;
        }
        if (length != 0) memcpy(m->run + used, bytes, (size_t)length);
        used += length;
        m->offsets[i + 1] = used;
    }
    return 0;
}

/* Reads an array of numbers. The elements of a type that a getter gives get a copy of
 * their own. The elements of any other type move the cursor only. Returns 0 or 2. */
static int number_array(aotx_modelfile *f, aotx_meta *m, unsigned width)
{
    uint64_t span;
    uint64_t i;
    int keep_whole = (m->element_type == AOTX_GGUF_I32 || m->element_type == AOTX_GGUF_U32);
    int keep_real = (m->element_type == AOTX_GGUF_F32);
    int keep_bool = (m->element_type == AOTX_GGUF_BOOL);
    int rc;
    if (m->count > UINT64_MAX / width) {
        return aotx_gguf_refuse(f, "an array is too large for the file");
    }
    span = m->count * width;
    /* The span check comes before the allocation, so a large count in a small file cannot
     * ask for memory. */
    rc = aotx_gguf_need(f, span);
    if (rc != 0) {
        return rc;
    }
    if ((keep_whole || keep_real) &&
        aotx_gguf_charge(f, (m->count + 1u) * sizeof(int32_t)) != 0) return 2;
    if (keep_whole) {
        m->i32 = (int32_t *)calloc((size_t)m->count + 1u, sizeof(int32_t));
        if (m->i32 == NULL) {
            return aotx_gguf_refuse(f, "the memory for an array is not there");
        }
    }
    if (keep_real) {
        m->f32 = (float *)calloc((size_t)m->count + 1u, sizeof(float));
        if (m->f32 == NULL) {
            return aotx_gguf_refuse(f, "the memory for an array is not there");
        }
    }
    if (keep_bool) {
        if (aotx_gguf_charge(f, m->count + 1u) != 0) return 2;
        m->run = (uint8_t *)calloc((size_t)m->count + 1u, 1u);
        if (m->run == NULL) return aotx_gguf_refuse(f, "the memory for a boolean array is not there");
    }
    for (i = 0; i < m->count; i++) {
        uint64_t raw = 0;
        rc = aotx_gguf_number(f, width, &raw);
        if (rc != 0) {
            return rc;
        }
        if (keep_whole && m->i32 != NULL) {
            int64_t value = is_signed(m->element_type) ? signed_value(raw, width) : (int64_t)raw;
            if (value > INT32_MAX || value < INT32_MIN) {
                /* An element of this array does not fit the type that the getter gives, so
                 * the array is not available. */
                free(m->i32);
                m->i32 = NULL;
            } else {
                m->i32[i] = (int32_t)value;
            }
        }
        if (keep_real) {
            m->f32[i] = float_of((uint32_t)raw);
        }
        if (keep_bool) {
            if (raw > 1u) return aotx_gguf_refuse(f, "a boolean array value is not 0 or 1");
            m->run[i] = (uint8_t)raw;
        }
    }
    return 0;
}

/* Reads an array value: the type of an element, the count, and the elements. */
static int array_value(aotx_modelfile *f, aotx_meta *m)
{
    uint64_t value = 0;
    unsigned width;
    int rc = aotx_gguf_number(f, 4, &value);
    if (rc != 0) {
        return rc;
    }
    m->element_type = (uint32_t)value;
    rc = aotx_gguf_number(f, 8, &m->count);
    if (rc != 0) {
        return rc;
    }
    if (m->element_type == AOTX_GGUF_ARRAY) {
        return aotx_gguf_refuse(f, "an array holds an array, which the format does not allow");
    }
    if (m->count > AOTX_GGUF_ARRAY_LIMIT) {
        return aotx_gguf_refuse(f, "an array holds more elements than the limit");
    }
    if (m->element_type == AOTX_GGUF_STRING) {
        return string_array(f, m);
    }
    width = width_of(m->element_type);
    if (width == 0) {
        return aotx_gguf_refuse(f, "an array holds a type that is not known");
    }
    return number_array(f, m, width);
}

int aotx_gguf_metadata(aotx_modelfile *f)
{
    uint64_t i;
    if (f->meta_count > 0) {
        if (aotx_gguf_charge(f, f->meta_count * sizeof(aotx_meta)) != 0) return 2;
        f->meta = (aotx_meta *)calloc((size_t)f->meta_count, sizeof(aotx_meta));
        if (f->meta == NULL) {
            return aotx_gguf_refuse(f, "the memory for the metadata is not there");
        }
    }
    for (i = 0; i < f->meta_count; i++) {
        aotx_meta *m = &f->meta[i];
        const unsigned char *key = NULL;
        uint64_t key_bytes = 0;
        uint64_t type = 0;
        int rc = aotx_gguf_text(f, &key, &key_bytes);
        if (rc != 0) {
            return rc;
        }
        if (key_bytes >= AOTX_GGUF_KEY_BYTES) {
            return aotx_gguf_refuse(f, "a metadata key is too long");
        }
        if (memchr(key, 0, (size_t)key_bytes) != NULL) {
            return aotx_gguf_refuse(f, "a metadata key holds a zero byte");
        }
        if (aotx_gguf_charge(f, key_bytes + 1u) != 0) return 2;
        m->key = (char *)malloc((size_t)key_bytes + 1u);
        if (m->key == NULL) {
            return aotx_gguf_refuse(f, "the memory for a metadata key is not there");
        }
        memcpy(m->key, key, (size_t)key_bytes);
        m->key[key_bytes] = '\0';
        rc = aotx_gguf_number(f, 4, &type);
        if (rc != 0) {
            return rc;
        }
        m->type = (uint32_t)type;
        if (m->type == AOTX_GGUF_ARRAY) {
            rc = array_value(f, m);
        } else if (m->type == AOTX_GGUF_STRING) {
            rc = string_value(f, m);
        } else {
            rc = scalar(f, m, m->type);
        }
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

void aotx_gguf_metadata_release(aotx_modelfile *f)
{
    uint64_t i;
    for (i = 0; i < f->meta_count && f->meta != NULL; i++) {
        free(f->meta[i].key);
        free(f->meta[i].text);
        free(f->meta[i].run);
        free(f->meta[i].offsets);
        free(f->meta[i].i32);
        free(f->meta[i].f32);
    }
    free(f->meta);
    f->meta = NULL;
    f->meta_count = 0;
}

static const aotx_meta *find(const aotx_modelfile *file, const char *key)
{
    uint64_t i;
    if (file == NULL || key == NULL) {
        return NULL;
    }
    for (i = 0; i < file->meta_count && file->meta != NULL; i++) {
        if (file->meta[i].key != NULL && strcmp(file->meta[i].key, key) == 0) {
            return &file->meta[i];
        }
    }
    return NULL;
}

/* Gives the value of any whole number type as a value without a sign. A value with a sign
 * that is below zero has no value without a sign, so the getter refuses it. */
static int whole(const aotx_meta *m, uint64_t *out)
{
    if (width_of(m->type) == 0 || m->type == AOTX_GGUF_F32 || m->type == AOTX_GGUF_F64) {
        return AOTX_META_TYPE;
    }
    if (is_signed(m->type)) {
        if (m->i < 0) {
            return AOTX_META_TYPE;
        }
        *out = (uint64_t)m->i;
        return 0;
    }
    *out = m->u;
    return 0;
}

int aotx_modelfile_u64(const aotx_modelfile *file, const char *key, uint64_t *value)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || value == NULL) {
        return AOTX_META_MISSING;
    }
    return whole(m, value);
}

int aotx_modelfile_u32(const aotx_modelfile *file, const char *key, uint32_t *value)
{
    const aotx_meta *m = find(file, key);
    uint64_t wide = 0;
    int rc;
    if (m == NULL || value == NULL) {
        return AOTX_META_MISSING;
    }
    rc = whole(m, &wide);
    if (rc != 0) {
        return rc;
    }
    if (wide > 0xffffffffu) {
        return AOTX_META_TYPE;
    }
    *value = (uint32_t)wide;
    return 0;
}

/* A value of type F64 becomes a value of type F32, which can lose low bits. */
int aotx_modelfile_f32(const aotx_modelfile *file, const char *key, float *value)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || value == NULL) {
        return AOTX_META_MISSING;
    }
    if (m->type != AOTX_GGUF_F32 && m->type != AOTX_GGUF_F64) {
        return AOTX_META_TYPE;
    }
    *value = (float)m->f;
    return 0;
}

int aotx_modelfile_string(const aotx_modelfile *file, const char *key, const char **value,
                          size_t *length)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || value == NULL) {
        return AOTX_META_MISSING;
    }
    if (m->type != AOTX_GGUF_STRING || m->text == NULL) {
        return AOTX_META_TYPE;
    }
    *value = m->text;
    if (length != NULL) {
        *length = (size_t)m->text_bytes;
    }
    return 0;
}

int aotx_modelfile_strings(const aotx_modelfile *file, const char *key, aotx_string_array *array)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || array == NULL) {
        return AOTX_META_MISSING;
    }
    if (m->type != AOTX_GGUF_ARRAY || m->element_type != AOTX_GGUF_STRING ||
        m->offsets == NULL) {
        return AOTX_META_TYPE;
    }
    array->count = m->count;
    array->bytes = m->run;
    array->offsets = m->offsets;
    return 0;
}

int aotx_modelfile_i32s(const aotx_modelfile *file, const char *key, const int32_t **values,
                        uint64_t *count)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || values == NULL) {
        return AOTX_META_MISSING;
    }
    if (m->type != AOTX_GGUF_ARRAY || m->i32 == NULL) {
        return AOTX_META_TYPE;
    }
    *values = m->i32;
    if (count != NULL) {
        *count = m->count;
    }
    return 0;
}

int aotx_modelfile_f32s(const aotx_modelfile *file, const char *key, const float **values,
                        uint64_t *count)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || values == NULL) {
        return AOTX_META_MISSING;
    }
    if (m->type != AOTX_GGUF_ARRAY || m->element_type != AOTX_GGUF_F32 || m->f32 == NULL) {
        return AOTX_META_TYPE;
    }
    *values = m->f32;
    if (count != NULL) {
        *count = m->count;
    }
    return 0;
}

int aotx_modelfile_bools(const aotx_modelfile *file, const char *key, const uint8_t **values,
                         uint64_t *count)
{
    const aotx_meta *m = find(file, key);
    if (m == NULL || values == NULL) return AOTX_META_MISSING;
    if (m->type != AOTX_GGUF_ARRAY || m->element_type != AOTX_GGUF_BOOL || m->run == NULL)
        return AOTX_META_TYPE;
    *values = m->run;
    if (count != NULL) *count = m->count;
    return 0;
}
