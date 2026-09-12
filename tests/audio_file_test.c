/* Purpose: Check audio component metadata and every trained tensor admission guard.
 * Owns: Distinct file readers, sparse test files and reversible parsed records.
 * Threading: One test process, no device dependency.
 * Lifetime: One file validation test. */
#include "disk/modelfile/audio.h"
#include "disk/modelfile/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned checks, failures;
static void check(int good, const char *what)
{
    ++checks;
    if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", what); }
}
static aotx_meta *find(aotx_modelfile *f, const char *name)
{
    for (uint64_t i = 0; i < f->meta_count; ++i)
        if (!strcmp(f->meta[i].key, name)) return &f->meta[i];
    return NULL;
}
static void refused(aotx_modelfile *f)
{
    aotx_audio_desc out, original;
    memset(&out, 0xa5, sizeof(out)); original = out;
    check(aotx_audio_file(f, &out) == 2, "invalid component is refused");
    check(memcmp(&out, &original, sizeof(out)) == 0, "refusal leaves the descriptor unchanged");
}

/* Keep trained shapes and metadata, but give each file distinct aligned tensor offsets. */
static aotx_modelfile *separate_file(const aotx_modelfile *source, unsigned item)
{
    uint64_t shift = (uint64_t)(item + 1u)*source->alignment;
    uint64_t table_bytes = 0;
    unsigned char *head = malloc((size_t)source->data_offset);
    aotx_modelfile *file = NULL;
    if (!head) return NULL;
    if (pread(source->fd, head, (size_t)source->data_offset, 0) != (ssize_t)source->data_offset) {
        free(head); return NULL;
    }
    for (uint64_t t = 0; t < source->tensor_count; ++t)
        table_bytes += 24u + strlen(source->tensors[t].name) + 8u*source->tensors[t].dim_count;
    if (table_bytes > source->pos) { free(head); return NULL; }
    uint64_t cursor = source->pos - table_bytes;
    for (uint64_t t = 0; t < source->tensor_count; ++t) {
        const aotx_tensor_info *info = &source->tensors[t];
        cursor += 16u + strlen(info->name) + 8u*info->dim_count;
        uint64_t original = 0;
        for (unsigned b = 0; b < 8; ++b) original |= (uint64_t)head[cursor+b] << (8u*b);
        if (original != info->offset) { free(head); return NULL; }
        for (unsigned b = 0; b < 8; ++b) head[cursor+b] = (unsigned char)((original+shift) >> (8u*b));
        cursor += 8u;
    }
    char path[] = "/tmp/aotx-audio-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { free(head); return NULL; }
    unsigned char tag = (unsigned char)(item + 1u);
    int good = ftruncate(fd, (off_t)(source->file_bytes+shift)) == 0 &&
        pwrite(fd, head, (size_t)source->data_offset, 0) == (ssize_t)source->data_offset &&
        pwrite(fd, &tag, 1, (off_t)(source->data_offset+shift)) == 1;
    if (good && aotx_modelfile_open(path, &file)) file = NULL;
    close(fd); unlink(path); free(head);
    return file;
}

static void descriptor_matches(aotx_modelfile *file, const aotx_audio_desc *base,
                               uint64_t shift, unsigned item)
{
    aotx_audio_desc out;
    memset(&out, 0xa5, sizeof(out));
    int rc = aotx_audio_file(file, &out);
    check(rc == 0, "each independent file is accepted");
    if (rc) return;
    check(out.bytes == base->bytes + shift, "the descriptor keeps its own file extent");
    for (unsigned t = 0; t < AOTX_AUDIO_BASE_TENSORS; ++t)
        check(out.base[t] == base->base[t] + shift, "base tensor offsets belong to the current file");
    for (unsigned l = 0; l < AOTX_AUDIO_LAYERS; ++l)
        for (unsigned t = 0; t < AOTX_AUDIO_LAYER_TENSORS; ++t)
            check(out.layer[l][t] == base->layer[l][t] + shift, "layer offsets belong to the current file");
    unsigned char tag = 0;
    check(aotx_modelfile_read(file, shift, 1, &tag) == 0 && tag == item+1u,
          "a read uses the current file descriptor");
}

static void file_batch(const aotx_modelfile *source, const aotx_audio_desc *base, unsigned count)
{
    aotx_modelfile **files = calloc(count, sizeof(*files));
    check(files != NULL, "the requested file batch is allocated");
    if (!files) return;
    unsigned opened = 0;
    for (unsigned i = 0; i < count; ++i) {
        files[i] = separate_file(source, i);
        if (files[i]) ++opened;
    }
    check(opened == count, "every requested distinct file opens");
    if (opened == count) {
        for (unsigned pass = 0; pass < 3; ++pass) {
            for (unsigned n = 0; n < count; ++n) {
                unsigned i = pass == 1 ? count-1u-n : (n*17u)%count;
                unsigned t = (i*37u)%489u;
                uint32_t saved = files[i]->tensors[t].type;
                if (pass == 1) files[i]->tensors[t].type = AOTX_TENSOR_Q8_0;
                if (pass == 1) refused(files[i]);
                else descriptor_matches(files[i], base, (uint64_t)(i+1u)*source->alignment, i);
                if (pass == 1 && count > 1) {
                    unsigned next = (i+1u)%count;
                    descriptor_matches(files[next], base, (uint64_t)(next+1u)*source->alignment, next);
                }
                files[i]->tensors[t].type = saved;
            }
        }
        for (unsigned i = 0; i < count; i += 2) {
            aotx_modelfile_close(files[i]); files[i] = NULL;
        }
        for (unsigned i = 1; i < count; i += 2)
            descriptor_matches(files[i], base, (uint64_t)(i+1u)*source->alignment, i);
    }
    for (unsigned i = 0; i < count; ++i) aotx_modelfile_close(files[i]);
    free(files);
}
int main(int argc, char **argv)
{
    aotx_modelfile *f = NULL;
    aotx_audio_desc descriptor;
    if (argc != 2) return 2;
    if (aotx_modelfile_open(argv[1], &f)) return 1;
    check(aotx_audio_file(f, &descriptor) == 0, "complete trained component is accepted");
    if (failures) return 1;
    for (uint64_t i = 0; i < f->tensor_count; ++i) {
        aotx_tensor_info saved = f->tensors[i];
        f->tensors[i].type = 8; refused(f); f->tensors[i] = saved;
        ++f->tensors[i].dims[0]; refused(f); f->tensors[i] = saved;
        ++f->tensors[i].offset; refused(f); f->tensors[i] = saved;
        f->tensors[i].offset = f->tensors[(i+1)%f->tensor_count].offset;
        refused(f); f->tensors[i] = saved;
    }
    --f->tensor_count; refused(f); f->tensor_count += 2; refused(f); --f->tensor_count;
    const char *keys[] = {"general.architecture", "general.type", "clip.projector_type",
        "clip.has_audio_encoder", "clip.audio.projection_dim", "clip.audio.embedding_length",
        "clip.audio.feed_forward_length", "clip.audio.block_count", "clip.audio.attention.head_count",
        "clip.audio.num_mel_bins", "clip.audio.attention.layer_norm_epsilon"};
    for (unsigned i=0;i<sizeof keys/sizeof keys[0];++i) {
        aotx_meta *m = find(f,keys[i]);check(m != NULL,"required metadata is present");
        if (!m) continue;
        uint32_t type=m->type;m->type=UINT32_MAX;refused(f);m->type=type;
    }
    aotx_meta *epsilon = find(f,"clip.audio.attention.layer_norm_epsilon");
    double value=epsilon->f;epsilon->f=1e-6;refused(f);epsilon->f=value;
    char *key = f->meta[1].key; f->meta[1].key = f->meta[0].key;
    refused(f); f->meta[1].key = key;
    check(aotx_audio_file(f, &descriptor) == 0, "restored component still passes");
    file_batch(f, &descriptor, 1);
    file_batch(f, &descriptor, 64);
    aotx_modelfile_close(f);
    printf("checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
