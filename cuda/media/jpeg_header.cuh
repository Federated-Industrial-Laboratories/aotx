/* Purpose: Validate JPEG markers, component layouts and scan progression on the device.
 * Owns: Quantization tables, Huffman tables and component state of each image.
 * Launch shape: One image thread processes one bounded marker per step.
 * Lifetime: One JPEG image generation. */
#ifndef AOTX_MEDIA_JPEG_HEADER_CUH
#define AOTX_MEDIA_JPEG_HEADER_CUH
#include "media/jpeg.cuh"

static __device__ __forceinline__ bool aotx_jpeg_quant(aotx_image_job *j, const unsigned char *p, uint32_t n) {
    while (n) {
        uint32_t table = *p & 15u, precision = *p >> 4;
        if (precision) { aotx_image_refuse(j, AOTX_IMAGE_UNSUPPORTED); return false; }
        if (table >= 4 || n < 65) return false;
        for (uint32_t i = 0; i < 64; ++i) {
            if (!p[i + 1]) return false;
            j->quant[table][aotx_jpeg_order[i]] = p[i + 1];
        }
        j->quant_valid[table] = 1; p += 65; n -= 65;
    }
    return true;
}
static __device__ __forceinline__ bool aotx_jpeg_tables(aotx_image_job *j, const unsigned char *p, uint32_t n) {
    while (n) {
        uint32_t kind = *p >> 4, table = *p & 15u;
        if (n < 17 || kind > 1 || table >= 4) return false;
        aotx_jpeg_huffman *h = &j->huffman[kind][table];
        uint32_t total = 0, code = 0;
        for (uint32_t bits = 1; bits <= 16; ++bits) {
            uint32_t entries = p[bits];
            if (code + entries >= (1u << bits) && entries) return false;
            h->first[bits] = (uint16_t)code;
            h->start[bits] = (uint16_t)total;
            h->count[bits] = (uint16_t)entries;
            total += entries;
            code = (code + entries) << 1;
        }
        if (!total || total > 256 || n < 17 + total) return false;
        for (uint32_t i = 0; i < total; ++i) h->symbol[i] = p[17 + i];
        h->valid = 1; p += 17 + total; n -= 17 + total;
    }
    return true;
}
static __device__ __forceinline__ bool aotx_jpeg_frame(aotx_image_job *j, const unsigned char *p,
                                       uint32_t n, uint32_t marker) {
    if (j->frame || n < 6) return false;
    if (p[0] != 8 || (p[5] != 1 && p[5] != 3)) {
        aotx_image_refuse(j, AOTX_IMAGE_UNSUPPORTED); return false;
    }
    j->height = aotx_jpeg_u16(p + 1); j->width = aotx_jpeg_u16(p + 3);
    j->components = p[5];
    if (n != 6 + 3 * j->components || !aotx_jpeg_dimensions(j)) return false;
    j->hmax = j->vmax = 1;
    for (uint32_t i = 0; i < j->components; ++i) {
        aotx_jpeg_component *c = j->channel + i;
        c->id = p[6 + i * 3]; c->h = p[7 + i * 3] >> 4;
        c->v = p[7 + i * 3] & 15u; c->quant = p[8 + i * 3];
        if (!c->h || !c->v || c->quant >= 4) return false;
        if (c->h > 2 || c->v > 2 || (i && (c->h != 1 || c->v != 1)) ||
            (j->components == 1 && (c->h != 1 || c->v != 1)) ||
            (j->components == 3 && c->id != i + 1) || (c->h == 1 && c->v == 2)) {
            aotx_image_refuse(j, AOTX_IMAGE_UNSUPPORTED); return false;
        }
        for (uint32_t k = 0; k < i; ++k) if (j->channel[k].id == c->id) return false;
        for (uint32_t k = 0; k < 64; ++k) c->approximation[k] = 255;
        j->hmax = max(j->hmax, c->h); j->vmax = max(j->vmax, c->v);
        c->predictor = 0; c->quant_bound = 0;
    }
    j->mcu_cols = aotx_jpeg_ceil(j->width, j->hmax * 8);
    j->mcu_rows = aotx_jpeg_ceil(j->height, j->vmax * 8);
    uint64_t blocks = 0;
    for (uint32_t i = 0; i < j->components; ++i) {
        aotx_jpeg_component *c = j->channel + i;
        c->width = aotx_jpeg_ceil(j->width * c->h, j->hmax);
        c->height = aotx_jpeg_ceil(j->height * c->v, j->vmax);
        c->cols = aotx_jpeg_ceil(c->width, 8); c->rows = aotx_jpeg_ceil(c->height, 8);
        c->stride = j->mcu_cols * c->h; c->base = (uint32_t)blocks;
        blocks += (uint64_t)c->stride * j->mcu_rows * c->v;
    }
    if (blocks > 0xFFFFFFFFu || blocks > j->coefficient_count / 64 ||
        blocks > j->plane_bytes / 64) {
        aotx_image_refuse(j, AOTX_IMAGE_LIMIT); return false;
    }
    j->blocks = (uint32_t)blocks; j->progressive = marker == 194;
    j->frame = 1; j->phase = AOTX_IMAGE_ZERO;
    return true;
}
static __device__ __forceinline__ bool aotx_jpeg_scan_header(aotx_image_job *j, const unsigned char *p, uint32_t n) {
    if (!j->frame || n < 4 || !p[0] || p[0] > j->components || n != 4u + 2u * p[0]) return false;
    if (j->components == 3 && !j->jfif) {
        aotx_image_refuse(j, AOTX_IMAGE_UNSUPPORTED); return false;
    }
    j->scan_count = p[0]; j->ss = p[1 + 2 * p[0]]; j->se = p[2 + 2 * p[0]];
    j->ah = p[3 + 2 * p[0]] >> 4; j->al = p[3 + 2 * p[0]] & 15u;
    if (j->ss > j->se || j->se > 63 || j->ah > 13 || j->al > 13) return false;
    if (!j->progressive && (j->ss || j->se != 63 || j->ah || j->al)) return false;
    if (j->progressive && ((!j->ss && j->se) || (j->ss && j->scan_count != 1) ||
        (j->ah && j->al + 1 != j->ah))) return false;
    for (uint32_t i = 0; i < j->scan_count; ++i) {
        uint32_t ci = 0;
        while (ci < j->components && j->channel[ci].id != p[1 + i * 2]) ++ci;
        if (ci == j->components) return false;
        for (uint32_t k = 0; k < i; ++k) if (j->component[k] == ci) return false;
        j->component[i] = ci;
        uint32_t dc = p[2 + i * 2] >> 4, ac = p[2 + i * 2] & 15u;
        if (dc >= 4 || ac >= 4 || (!j->ss && !j->ah && !j->huffman[0][dc].valid) ||
            ((!j->progressive || j->ss) && !j->huffman[1][ac].valid)) return false;
        j->dc_table[i] = dc; j->ac_table[i] = ac;
        aotx_jpeg_component *c = j->channel + ci;
        if (!j->quant_valid[c->quant] || (j->ss && c->approximation[0] == 255)) return false;
        if (!c->quant_bound) {
            for (uint32_t k = 0; k < 64; ++k) c->quant_values[k] = j->quant[c->quant][k];
            c->quant_bound = 1;
        } else {
            for (uint32_t k = 0; k < 64; ++k)
                if (c->quant_values[k] != j->quant[c->quant][k]) return false;
        }
        for (uint32_t k = j->ss; k <= j->se; ++k) {
            if ((!j->ah && c->approximation[k] != 255) ||
                (j->ah && c->approximation[k] != j->ah)) return false;
            c->approximation[k] = (unsigned char)j->al;
        }
        c->predictor = 0;
    }
    const aotx_jpeg_component *c = j->channel + j->component[0];
    j->units = j->scan_count == 1 ? c->cols * c->rows : j->mcu_cols * j->mcu_rows;
    j->unit = j->eob = j->bit_count = j->restart_next = 0;
    ++j->scans; j->phase = AOTX_IMAGE_SCAN;
    return true;
}
static __device__ __forceinline__ void aotx_jpeg_header(aotx_image_job *j) {
    if (!aotx_jpeg_extent(j, 2)) return;
    if (j->source[j->cursor++] != 255) { aotx_image_refuse(j, AOTX_IMAGE_INVALID); return; }
    uint32_t marker = j->source[j->cursor++];
    if (marker == 255) { --j->cursor; return; }
    if (marker == 217) {
        bool valid = j->frame && j->scans && j->cursor == j->bytes;
        for (uint32_t i = 0; i < j->components; ++i)
            valid = valid && j->channel[i].quant_bound && j->channel[i].approximation[0] != 255;
        if (valid) j->phase = AOTX_IMAGE_IDCT;
        else aotx_image_refuse(j, AOTX_IMAGE_INVALID);
        return;
    }
    if (!marker || marker == 216 || (marker >= 208 && marker <= 215) || marker == 1 ||
        !aotx_jpeg_extent(j, 2)) { aotx_image_refuse(j, AOTX_IMAGE_INVALID); return; }
    uint32_t bytes = aotx_jpeg_u16(j->source + j->cursor);
    if (bytes < 2 || !aotx_jpeg_extent(j, bytes)) { aotx_image_refuse(j, AOTX_IMAGE_INVALID); return; }
    const unsigned char *p = j->source + j->cursor + 2;
    uint32_t n = bytes - 2; j->cursor += bytes;
    bool valid = true;
    if (marker == 219) valid = aotx_jpeg_quant(j, p, n);
    else if (marker == 196) valid = aotx_jpeg_tables(j, p, n);
    else if (marker == 192 || marker == 194) valid = aotx_jpeg_frame(j, p, n, marker);
    else if (marker == 218) valid = aotx_jpeg_scan_header(j, p, n);
    else if (marker == 221) {
        valid = n == 2;
        if (valid) j->restart_interval = aotx_jpeg_u16(p);
    } else if (marker == 224 && n >= 5 && p[0] == 'J' && p[1] == 'F' &&
               p[2] == 'I' && p[3] == 'F' && !p[4]) {
        valid = n >= 14 && p[5] == 1 && p[6] <= 2 &&
            aotx_jpeg_u16(p + 8) && aotx_jpeg_u16(p + 8) == aotx_jpeg_u16(p + 10) &&
            n == 14u + 3u * p[12] * p[13];
        if (valid) j->jfif = 1;
    } else if (marker == 225 || marker == 226 || marker == 238 ||
               (marker >= 192 && marker <= 207)) {
        aotx_image_refuse(j, AOTX_IMAGE_UNSUPPORTED); return;
    } else if (!((marker >= 224 && marker <= 239) || marker == 254)) valid = false;
    if (!valid && j->phase != AOTX_IMAGE_REFUSED) aotx_image_refuse(j, AOTX_IMAGE_INVALID);
}
#endif
