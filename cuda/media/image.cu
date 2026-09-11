/* Purpose: Advance bounded image batches and initialize coefficient storage.
 * Owns: Decoder phase transitions and cancellation results.
 * Launch shape: Threads stride image jobs; clear blocks stride coefficient arrays.
 * Lifetime: One image transfer and its finite decode steps. */
#include "media/jpeg.cuh"
#include "media/jpeg_header.cuh"
#include "media/jpeg_entropy.cuh"
__global__ void aotx_image_step(aotx_image_job *jobs, uint32_t count, uint32_t quantum) {
    for (uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
         at < count; at += blockDim.x * gridDim.x) {
        aotx_image_job *j = jobs + at;
        if (j->phase == AOTX_IMAGE_READY || j->phase == AOTX_IMAGE_REFUSED) continue;
        if (j->cancel) { aotx_image_refuse(j, AOTX_IMAGE_CANCELLED); continue; }
        if (j->phase == AOTX_IMAGE_NEW) {
            j->status = j->frame = j->scans = j->jfif = j->restart_interval = 0;
            for (uint32_t k = 0; k < 4; ++k) {
                j->quant_valid[k] = j->huffman[0][k].valid = j->huffman[1][k].valid = 0;
            }
            if (!j->source || !j->rgb || !j->bytes || !quantum) {
                aotx_image_refuse(j, AOTX_IMAGE_INVALID); continue;
            }
            if (j->format == AOTX_IMAGE_RGB8) {
                if (aotx_jpeg_dimensions(j)) {
                    if (j->bytes != (uint64_t)j->width * j->height * 3)
                        aotx_image_refuse(j, AOTX_IMAGE_INVALID);
                    else j->phase = AOTX_IMAGE_RAW;
                }
                continue;
            }
            if (j->format != AOTX_IMAGE_JPEG) {
                aotx_image_refuse(j, AOTX_IMAGE_UNSUPPORTED); continue;
            }
            if (j->bytes < 4 || j->source[0] != 255 || j->source[1] != 216 ||
                !j->coefficients || !j->planes) {
                aotx_image_refuse(j, AOTX_IMAGE_INVALID); continue;
            }
            j->cursor = 2; j->phase = AOTX_IMAGE_HEADER;
        }
        if (j->phase == AOTX_IMAGE_HEADER) aotx_jpeg_header(j);
        else if (j->phase == AOTX_IMAGE_SCAN) aotx_jpeg_scan(j, quantum);
    }
}
__global__ void aotx_image_zero(aotx_image_job *jobs, uint32_t count) {
    for (uint32_t at = blockIdx.y; at < count; at += gridDim.y) {
        aotx_image_job *j = jobs + at;
        if (j->phase != AOTX_IMAGE_ZERO) continue;
        uint64_t values = (uint64_t)j->blocks * 64;
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
             i < values; i += (uint64_t)blockDim.x * gridDim.x) j->coefficients[i] = 0;
    }
}
__global__ void aotx_image_finish(aotx_image_job *jobs, uint32_t count) {
    for (uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
         at < count; at += blockDim.x * gridDim.x) {
        aotx_image_job *j = jobs + at;
        if (j->phase == AOTX_IMAGE_ZERO) j->phase = AOTX_IMAGE_HEADER;
        else if (j->phase == AOTX_IMAGE_IDCT || j->phase == AOTX_IMAGE_RAW)
            j->phase = AOTX_IMAGE_READY;
    }
}
