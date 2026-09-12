/* Purpose: Produce image fixtures and independent JPEG coefficient and pixel references.
 * Owns: Test-only codec instances and their complete output batches.
 * Threading: One test thread prepares references before device execution.
 * Lifetime: One test process; no host codec enters the product runtime. */
#ifndef AOTX_IMAGE_REFERENCE_H
#define AOTX_IMAGE_REFERENCE_H
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <vector>
#include <string>

struct aotx_image_reference {
    std::vector<unsigned char> bytes, rgb, planes[3];
    std::vector<int32_t> coefficients[3];
    unsigned width = 0, height = 0, components = 0, cols[3] = {}, rows[3] = {};
    unsigned cw[3] = {}, ch[3] = {};
};
static std::vector<unsigned char> aotx_image_encode(unsigned seed, unsigned width, unsigned height,
                                                   unsigned sampling, bool progressive, unsigned restart) {
    jpeg_compress_struct c = {}; jpeg_error_mgr error = {};
    c.err = jpeg_std_error(&error); jpeg_create_compress(&c);
    unsigned char *encoded = nullptr; unsigned long bytes = 0;
    jpeg_mem_dest(&c, &encoded, &bytes);
    c.image_width = width; c.image_height = height;
    c.input_components = sampling == 3 ? 1 : 3;
    c.in_color_space = sampling == 3 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&c); jpeg_set_quality(&c, 83, TRUE);
    c.comp_info[0].h_samp_factor = sampling == 0 || sampling == 3 ? 1 : 2;
    c.comp_info[0].v_samp_factor = sampling == 2 ? 2 : 1;
    c.restart_interval = restart;
    if (progressive) jpeg_simple_progression(&c);
    jpeg_start_compress(&c, TRUE);
    std::vector<unsigned char> row(width * c.input_components);
    while (c.next_scanline < height) {
        unsigned y = c.next_scanline;
        for (unsigned x = 0; x < width; ++x)
            for (unsigned k = 0; k < (unsigned)c.input_components; ++k)
                row[x * c.input_components + k] = (unsigned char)((x * (7 + k * 11) + y * (19 + k) +
                    seed * 23 + ((x ^ y ^ seed) & 7) * 29) & 255);
        JSAMPROW p = row.data(); jpeg_write_scanlines(&c, &p, 1);
    }
    jpeg_finish_compress(&c);
    std::vector<unsigned char> result(encoded, encoded + bytes);
    free(encoded); jpeg_destroy_compress(&c);
    return result;
}
static void aotx_image_reference_open(jpeg_decompress_struct *c, jpeg_error_mgr *error,
                                      const std::vector<unsigned char> &bytes) {
    c->err = jpeg_std_error(error); jpeg_create_decompress(c);
    jpeg_mem_src(c, bytes.data(), bytes.size()); jpeg_read_header(c, TRUE);
    c->dct_method = JDCT_ISLOW; c->do_fancy_upsampling = TRUE;
}
static aotx_image_reference aotx_image_reference_read(std::vector<unsigned char> bytes) {
    aotx_image_reference out; out.bytes = std::move(bytes);
    jpeg_decompress_struct c = {}; jpeg_error_mgr error = {};
    aotx_image_reference_open(&c, &error, out.bytes);
    out.width = c.image_width; out.height = c.image_height; out.components = c.num_components;
    jvirt_barray_ptr *arrays = jpeg_read_coefficients(&c);
    for (unsigned i = 0; i < out.components; ++i) {
        const jpeg_component_info *component = c.comp_info + i;
        out.cols[i] = component->width_in_blocks; out.rows[i] = component->height_in_blocks;
        out.coefficients[i].resize((size_t)out.cols[i] * out.rows[i] * 64);
        for (unsigned y = 0; y < out.rows[i]; ++y) {
            JBLOCKARRAY row = c.mem->access_virt_barray((j_common_ptr)&c, arrays[i], y, 1, FALSE);
            for (unsigned x = 0; x < out.cols[i]; ++x)
                for (unsigned k = 0; k < 64; ++k)
                    out.coefficients[i][((size_t)y * out.cols[i] + x) * 64 + k] = row[0][x][k];
        }
    }
    jpeg_finish_decompress(&c); jpeg_destroy_decompress(&c);
    c = {}; error = {}; aotx_image_reference_open(&c, &error, out.bytes);
    c.out_color_space = JCS_RGB; jpeg_start_decompress(&c);
    out.rgb.resize((size_t)out.width * out.height * 3);
    while (c.output_scanline < c.output_height) {
        JSAMPROW row = out.rgb.data() + (size_t)c.output_scanline * out.width * 3;
        jpeg_read_scanlines(&c, &row, 1);
    }
    jpeg_finish_decompress(&c); jpeg_destroy_decompress(&c);
    c = {}; error = {}; aotx_image_reference_open(&c, &error, out.bytes);
    c.raw_data_out = TRUE; jpeg_start_decompress(&c);
    std::vector<unsigned char> raw[3]; std::vector<JSAMPROW> pointers[3]; JSAMPARRAY image[3] = {};
    for (unsigned i = 0; i < out.components; ++i) {
        const jpeg_component_info *component = c.comp_info + i;
        out.cw[i] = component->downsampled_width; out.ch[i] = component->downsampled_height;
        unsigned stride = out.cols[i] * 8, rows = component->v_samp_factor * 8;
        out.planes[i].resize((size_t)out.cw[i] * out.ch[i]);
        raw[i].resize((size_t)stride * rows); pointers[i].resize(rows);
        for (unsigned y = 0; y < rows; ++y) pointers[i][y] = raw[i].data() + (size_t)y * stride;
        image[i] = pointers[i].data();
    }
    unsigned group = 0;
    while (c.output_scanline < c.output_height) {
        jpeg_read_raw_data(&c, image, c.max_v_samp_factor * 8);
        for (unsigned i = 0; i < out.components; ++i) {
            unsigned rows = c.comp_info[i].v_samp_factor * 8;
            for (unsigned y = 0; y < rows && group * rows + y < out.ch[i]; ++y)
                for (unsigned x = 0; x < out.cw[i]; ++x)
                    out.planes[i][(size_t)(group * rows + y) * out.cw[i] + x] = image[i][y][x];
        }
        ++group;
    }
    jpeg_finish_decompress(&c); jpeg_destroy_decompress(&c);
    return out;
}
#endif
