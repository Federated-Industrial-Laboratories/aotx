/* Purpose: Decode sequential and progressive Huffman scans in finite device steps.
 * Owns: Bit cursors, predictors, EOB runs and quantized image coefficients.
 * Launch shape: One thread per image consumes a bounded number of complete MCUs.
 * Lifetime: One scan; state persists across graph launches. */
#ifndef AOTX_MEDIA_JPEG_ENTROPY_CUH
#define AOTX_MEDIA_JPEG_ENTROPY_CUH
#include "media/jpeg.cuh"

static __device__ __forceinline__ uint32_t aotx_jpeg_bits(aotx_image_job *j, uint32_t count) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < count && !j->status; ++i) {
        if (!j->bit_count) {
            if (!aotx_jpeg_extent(j, 1)) return 0;
            j->bit_byte = j->source[j->cursor++];
            if (j->bit_byte == 255) {
                if (!aotx_jpeg_extent(j, 1)) return 0;
                if (j->source[j->cursor++] != 0) {
                    aotx_image_refuse(j, AOTX_IMAGE_INVALID); return 0;
                }
            }
            j->bit_count = 8;
        }
        value = (value << 1) | ((j->bit_byte >> --j->bit_count) & 1u);
    }
    return value;
}
static __device__ __forceinline__ uint32_t aotx_jpeg_symbol(aotx_image_job *j, uint32_t kind, uint32_t table) {
    const aotx_jpeg_huffman *h = &j->huffman[kind][table];
    uint32_t code = 0;
    for (uint32_t bits = 1; bits <= 16 && !j->status; ++bits) {
        code = (code << 1) | aotx_jpeg_bits(j, 1);
        if (code >= h->first[bits] && code - h->first[bits] < h->count[bits])
            return h->symbol[h->start[bits] + code - h->first[bits]];
    }
    aotx_image_refuse(j, AOTX_IMAGE_INVALID); return 0;
}
static __device__ __forceinline__ int32_t aotx_jpeg_signed(aotx_image_job *j, uint32_t bits) {
    if (!bits) return 0;
    uint32_t value = aotx_jpeg_bits(j, bits);
    return value < (1u << (bits - 1)) ? (int32_t)value - (int32_t)((1u << bits) - 1u) : (int32_t)value;
}
static __device__ __forceinline__ bool aotx_jpeg_padding(aotx_image_job *j) {
    uint32_t mask = (1u << j->bit_count) - 1u;
    if ((j->bit_byte & mask) != mask) return false;
    j->bit_count = 0;
    return true;
}
static __device__ __forceinline__ bool aotx_jpeg_restart(aotx_image_job *j) {
    if (!aotx_jpeg_padding(j) || j->eob || !aotx_jpeg_extent(j, 2) ||
        j->source[j->cursor] != 255 || j->source[j->cursor + 1] != 208 + j->restart_next) return false;
    j->cursor += 2; j->restart_next = (j->restart_next + 1) & 7u;
    for (uint32_t i = 0; i < j->components; ++i) j->channel[i].predictor = 0;
    return true;
}
static __device__ __forceinline__ bool aotx_jpeg_dc(aotx_image_job *j, uint32_t scan, int32_t *block) {
    aotx_jpeg_component *c = j->channel + j->component[scan];
    if (j->ah) {
        block[0] = (int32_t)((uint32_t)block[0] | (aotx_jpeg_bits(j, 1) << j->al));
        return !j->status;
    }
    uint32_t bits = aotx_jpeg_symbol(j, 0, j->dc_table[scan]);
    if (j->status || bits > 11) return false;
    int64_t prediction = (int64_t)c->predictor + aotx_jpeg_signed(j, bits);
    int64_t value = prediction * (1ll << j->al);
    if (prediction < -2147483647ll || prediction > 2147483647ll ||
        value < -2147483647ll || value > 2147483647ll) return false;
    c->predictor = (int32_t)prediction; block[0] = (int32_t)value;
    return !j->status;
}
static __device__ __forceinline__ bool aotx_jpeg_ac_first(aotx_image_job *j, uint32_t scan, int32_t *block) {
    if (j->eob) { --j->eob; return true; }
    uint32_t k = j->progressive ? j->ss : 1;
    while (k <= j->se && !j->status) {
        uint32_t symbol = aotx_jpeg_symbol(j, 1, j->ac_table[scan]);
        uint32_t run = symbol >> 4, bits = symbol & 15u;
        if (j->status || bits > 10) return false;
        if (bits) {
            if (run > j->se - k) return false;
            k += run;
            block[aotx_jpeg_order[k++]] = aotx_jpeg_signed(j, bits) * (1 << j->al);
        } else if (run == 15) {
            if (16 > j->se + 1 - k) return false;
            k += 16;
        } else {
            if (!j->progressive && run) return false;
            j->eob = (1u << run) + aotx_jpeg_bits(j, run) - 1u;
            break;
        }
    }
    return !j->status;
}
static __device__ __forceinline__ bool aotx_jpeg_refine(aotx_image_job *j, int32_t *value) {
    uint32_t bit = aotx_jpeg_bits(j, 1);
    uint32_t magnitude = *value < 0 ? (uint32_t)(-(int64_t)*value) : (uint32_t)*value;
    if (bit && !(magnitude & (1u << j->al))) {
        int64_t next = (int64_t)*value + (*value < 0 ? -(1ll << j->al) : (1ll << j->al));
        if (next < -2147483647ll || next > 2147483647ll) return false;
        *value = (int32_t)next;
    }
    return !j->status;
}
static __device__ __forceinline__ bool aotx_jpeg_ac_refine(aotx_image_job *j, uint32_t scan, int32_t *block) {
    uint32_t k = j->ss;
    if (!j->eob) {
        while (k <= j->se && !j->status) {
            uint32_t symbol = aotx_jpeg_symbol(j, 1, j->ac_table[scan]);
            uint32_t zeros = symbol >> 4, size = symbol & 15u;
            if (j->status || size > 1) return false;
            int32_t add = 0;
            if (size) add = aotx_jpeg_bits(j, 1) ? (1 << j->al) : -(1 << j->al);
            else if (zeros != 15) {
                j->eob = (1u << zeros) + aotx_jpeg_bits(j, zeros);
                break;
            } else zeros = 16;
            bool placed = !size;
            while (k <= j->se) {
                int32_t *value = block + aotx_jpeg_order[k];
                if (*value) {
                    if (!aotx_jpeg_refine(j, value)) return false;
                } else if (zeros) --zeros;
                else if (size) { *value = add; placed = true; ++k; break; }
                ++k;
                if (!size && !zeros) break;
            }
            if (zeros || !placed) return false;
        }
    }
    if (j->eob) {
        for (; k <= j->se; ++k) {
            int32_t *value = block + aotx_jpeg_order[k];
            if (*value && !aotx_jpeg_refine(j, value)) return false;
        }
        --j->eob;
    }
    return !j->status;
}
static __device__ __forceinline__ bool aotx_jpeg_block(aotx_image_job *j, uint32_t scan, uint32_t index) {
    if (index >= j->blocks) return false;
    int32_t *block = j->coefficients + (uint64_t)index * 64;
    if (!j->ss && !aotx_jpeg_dc(j, scan, block)) return false;
    if (!j->progressive || j->ss) {
        if (j->ah) return aotx_jpeg_ac_refine(j, scan, block);
        return aotx_jpeg_ac_first(j, scan, block);
    }
    return true;
}
static __device__ __forceinline__ void aotx_jpeg_scan(aotx_image_job *j, uint32_t quantum) {
    uint32_t end = j->unit + min(quantum, j->units - j->unit);
    while (j->unit < end && !j->status) {
        if (j->restart_interval && j->unit && j->unit % j->restart_interval == 0 &&
            !aotx_jpeg_restart(j)) { aotx_image_refuse(j, AOTX_IMAGE_INVALID); return; }
        bool valid = true;
        if (j->scan_count == 1) {
            const aotx_jpeg_component *c = j->channel + j->component[0];
            uint32_t index = c->base + (j->unit / c->cols) * c->stride + j->unit % c->cols;
            valid = aotx_jpeg_block(j, 0, index);
        } else {
            uint32_t mx = j->unit % j->mcu_cols, my = j->unit / j->mcu_cols;
            for (uint32_t s = 0; s < j->scan_count && valid; ++s) {
                const aotx_jpeg_component *c = j->channel + j->component[s];
                for (uint32_t v = 0; v < c->v && valid; ++v)
                    for (uint32_t h = 0; h < c->h && valid; ++h)
                        valid = aotx_jpeg_block(j, s, c->base + (my * c->v + v) * c->stride + mx * c->h + h);
            }
        }
        ++j->unit;
        uint32_t remaining = j->units - j->unit;
        if (j->restart_interval)
            remaining = min(remaining, (j->restart_interval - j->unit % j->restart_interval) % j->restart_interval);
        if (!valid || j->eob > remaining) { aotx_image_refuse(j, AOTX_IMAGE_INVALID); return; }
    }
    if (j->unit == j->units && !j->status) {
        if (j->eob || !aotx_jpeg_padding(j)) aotx_image_refuse(j, AOTX_IMAGE_INVALID);
        else j->phase = AOTX_IMAGE_HEADER;
    }
}
#endif
