/* Purpose: Refuse invalid runtime references, incompatible files and changed startup generations.
 * Owns: Distinct data modules, prepared metadata and changed source generations.
 * Threading: One disk process checks complete batches at one and 64 modules.
 * Lifetime: All temporary files, reader handles and writer leases end with each case. */
#include "runtime_dependency_fixture.h"
#include "disk/runtime/activate.h"
#include "cognitive/checkpoint_io.h"
#include "profile/profile.cuh"

static void prepared(aotx_dependency_fixture *f) {
    const char *phrases = "I cannot\n";
    asset(f, "quality/refusal-phrases.txt", 1, phrases, strlen(phrases));
    aotx_ccir_put(f->index.header + 16, f->index.count, 4);
    f->input[3].section.bytes = AOTX_RUNTIME_HEADER + f->index.count * AOTX_RUNTIME_ROW;
#ifdef AOTX_AFFECT
    aotx_ccir_put(f->index.header + 20, AOTX_RUNTIME_AFFECT, 4);
#endif
    memcpy(f->memory, "AOTXOBJ1", 8); aotx_ccir_put(f->memory + 8, 1, 4);
    aotx_ccir_put(f->memory + 12, 256, 4);
    aotx_ccir_put(f->memory + 32, 1, 8); aotx_ccir_put(f->memory + 40, 1, 8);
    f->memory[48] = 73; aotx_ccir_put(f->memory + 80, 128, 8);
    memcpy(f->live, "AOTXLCP1", 8); aotx_ccir_put(f->live + 8, 1, 4);
    aotx_ccir_put(f->live + 12, AOTX_CP_ROW, 4); aotx_ccir_put(f->live + 24, 128, 8);
    f->live[32] = 73; aotx_ccir_put(f->live + 48, 1, 8); aotx_ccir_put(f->live + 56, 1, 8);
    aotx_ccir_put(f->live + 64, 1, 8); aotx_ccir_put(f->live + 72, 1, 8);
}
static void test(const char *root, unsigned n, unsigned changed) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); prepared(f);
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_runtime_boot boot;
    int rc = aotx_runtime_prepare(path, journal, 86, &boot); CHECK(!rc);
    if (rc) { unlink(path); free(f); return; }
    unsigned char *image = NULL; uint32_t bytes = 0;
    CHECK(!aotx_checkpoint_file_read(path, &image, &bytes));
    if (changed) {
        f->body[n - 1][0] = 'X';
        for (uint32_t i = 0; i < f->index.count; ++i) {
            aotx_ccir_input *in = f->input + i + 5;
            if (in->data == f->body[n - 1])
                aotx_ccir_hash(in->data, in->section.bytes, f->index.rows[i] + 32);
        }
        CHECK(!aotx_ccir_append(path, f->input, f->count, &meta, NULL));
    }
    aotx_checkpoint_ring ring = {0}; ring.boot = 75;
    aotx_checkpoint_disk disk = {0}; disk.view.fd = -1;
    disk.runtime = 1; disk.runtime_sequence = 100; disk.path = path; disk.journal = journal; disk.ring = &ring;
    memcpy(disk.runtime_revision, boot.revision, 32);
    if (image) {
        rc = aotx_checkpoint_file_write(&disk, image, bytes);
        CHECK(rc == (changed ? AOTX_CCIR_CHANGED : AOTX_CCIR_BUSY));
        CHECK(disk.runtime_verified == !changed);
        CHECK(disk.view.generation == (changed ? 2u : 1u));
        CHECK(!ring.consumed && !ring.generation);
        if (changed) {
            CHECK(aotx_checkpoint_file_write(&disk, image, bytes) == AOTX_CCIR_CHANGED);
            CHECK(!disk.runtime_verified && !ring.consumed);
        }
    }
    aotx_ccir_close(&disk.view); aotx_runtime_release(&boot); free(image); free(f);
    CHECK(!rmdir(journal)); CHECK(!unlink(path));
}
static void checkpoint_ack(const char *root, unsigned n) {
    char path[256]; snprintf(path, sizeof(path), "%s/ack.aotxccir", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f));
    aotx_checkpoint_ring *ring = calloc(1, AOTX_CP_RING_BYTES);
    CHECK(f && ring); if (!f || !ring) { free(f); free(ring); return; }
    make(f, n, 0); prepared(f);
    ring->magic = AOTX_CP_MAGIC; ring->layout = AOTX_CP_LAYOUT;
    ring->boot = 9070 + n; ring->slots = AOTX_MEMORY_SNAPSHOTS; ring->slot_bytes = AOTX_CP_SLOT_BYTES;
    aotx_checkpoint_disk disk = {0}; disk.view.fd = -1; disk.path = path; disk.ring = ring;
    for (unsigned i = 0; i <= n; ++i) {
        unsigned char prior[16] = {0};
        if (i == n) {
            memcpy(prior, disk.view.incarnation, 16);
            aotx_ccir_input inputs[AOTX_CCIR_SECTIONS] = {0};
            for (unsigned j = 0; j < disk.view.count; ++j) {
                inputs[j].section = disk.view.sections[j]; inputs[j].source = AOTX_CCIR_REUSE;
            }
            CHECK(!aotx_ccir_writer_append(&disk.view, inputs, disk.view.count, &disk.view.meta, NULL));
            CHECK(disk.view.generation > 1);
            CHECK(!aotx_ccir_writer_replace(&disk.view, path, inputs, disk.view.count, &disk.view.meta, NULL));
            CHECK(disk.view.generation == 1 && memcmp(prior, disk.view.incarnation, 16));
        } else {
            aotx_ccir_put(f->live + 48, i + 1, 8); aotx_ccir_put(f->live + 56, i + 11, 8);
            aotx_ccir_put(f->live + 64, i + 1, 8);
            aotx_ccir_put(f->memory + 32, i + 1, 8); aotx_ccir_put(f->memory + 40, i + 11, 8);
        }
        unsigned char *slot = (unsigned char *)(ring + 1) + (i % ring->slots) * ring->slot_bytes;
        aotx_ccir_put(slot, ring->boot, 8); aotx_ccir_put(slot + 8, i + 1, 8);
        aotx_ccir_put(slot + 16, 256, 8);
        memcpy(slot + AOTX_CP_SLOT_HEADER, f->live, 128);
        memcpy(slot + AOTX_CP_SLOT_HEADER + 128, f->memory, 128);
        ring->head = i + 1;
        CHECK(aotx_checkpoint_disk_pass(&disk) == 1);
        CHECK(ring->consumed == i + 1 && ring->ack_serial == 2u * (i + 1) && !ring->error);
        CHECK(ring->ack_boot == ring->boot && ring->generation == disk.view.generation);
        CHECK(ring->durable_sequence == aotx_ccir_u64(f->live + 48) &&
            ring->durable_revision == aotx_ccir_u64(f->live + 64) && !ring->reserved[1]);
        CHECK(!memcmp(ring->incarnation, disk.view.incarnation, 16) &&
            !memcmp(ring->commit_digest, disk.view.commit_digest, 32));
        if (i == n) CHECK(ring->generation == 1 && memcmp(ring->incarnation, prior, 16));
    }
    aotx_ccir_close(&disk.view); free(f); free(ring); CHECK(!unlink(path));
}
static void compatibility(const char *root, unsigned n) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    for (unsigned defect = 0; defect < 13; ++defect) {
        make(f, n, 0); prepared(f); unsigned char *h = f->index.header;
        if (defect == 0) aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) ^ AOTX_RUNTIME_AFFECT, 4);
        if (defect == 1) aotx_ccir_put(h + 24, AOTX_WIRE_LAYOUT + 1, 4);
        if (defect == 2) aotx_ccir_put(h + 28, AOTX_SLOTS + 1, 4);
        if (defect == 3) aotx_ccir_put(h + 32, AOTX_COG_OBJECTS + 1, 4);
        if (defect == 4) aotx_ccir_put(h + 40, (uint64_t)AOTX_COG_PAYLOAD + 1, 8);
        if (defect == 5) aotx_ccir_put(h + 36, 87, 4);
        if (defect >= 6) {
            aotx_runtime_shared_profile profile;
            aotx_runtime_shared_current(&profile); aotx_runtime_shared_write(h, &profile);
            f->input[3].section.schema = 2;
            unsigned offset = 160 + 4 * (defect - 6);
            aotx_ccir_put(h + offset, aotx_ccir_u32(h + offset) + n, 4);
        }
        unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
        CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
        aotx_runtime_boot boot;
        CHECK(aotx_runtime_prepare(path, journal, 86, &boot) == AOTX_CCIR_UNSUPPORTED);
        CHECK(!boot.owned && !boot.root[0]);
        CHECK(access(journal, F_OK) != 0);
        aotx_runtime_release(&boot); rmdir(journal); CHECK(!unlink(path));
    }
    free(f);
}
static void shared_promotion(const char *root, unsigned n) {
    char path[256]; snprintf(path, sizeof(path), "%s/promote.aotxccir", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); prepared(f);
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_ccir_view before, after;
    CHECK(!aotx_ccir_open(path, NULL, &before)); aotx_ccir_close(&before);
    CHECK(!aotx_runtime_promote_shared(path));
    CHECK(!aotx_ccir_open(path, NULL, &after));
    CHECK(after.generation == before.generation + 1 && after.count == before.count);
    CHECK(!memcmp(before.incarnation, after.incarnation, 16) && !memcmp(before.lineage, after.lineage, 16));
    for (unsigned i = 0; i < before.count; ++i) {
        if (before.sections[i].type == AOTX_CCIR_RUNTIME) continue;
        CHECK(before.sections[i].offset == after.sections[i].offset &&
            before.sections[i].bytes == after.sections[i].bytes &&
            !memcmp(before.sections[i].digest, after.sections[i].digest, 32));
    }
    aotx_runtime_index *index = malloc(sizeof(*index)); CHECK(index != NULL);
    if (index) {
        CHECK(!aotx_runtime_index_read(after.fd, &after, index));
        CHECK((aotx_ccir_u32(index->header + 20) & AOTX_RUNTIME_SHARED) != 0); free(index);
    }
    uint64_t generation = after.generation;
    aotx_ccir_close(&after);
    CHECK(!aotx_runtime_promote_shared(path));
    CHECK(!aotx_ccir_open(path, NULL, &after)); CHECK(after.generation == generation);
    aotx_ccir_close(&after); CHECK(!unlink(path)); free(f);
}
static void shared_activation(const char *root, unsigned n) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/shared.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/shared-journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); prepared(f);
    aotx_runtime_shared_profile profile;
    aotx_runtime_shared_current(&profile); aotx_runtime_shared_write(f->index.header, &profile);
    f->input[3].section.schema = 2;
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_runtime_boot boot;
    int rc = aotx_runtime_prepare(path, journal, 86, &boot); CHECK(!rc);
    if (!rc) {
        CHECK((boot.features & AOTX_RUNTIME_SHARED) && boot.mode == 1);
        CHECK(!memcmp(&boot.shared, &profile, sizeof(profile)));
        aotx_runtime_release(&boot); CHECK(!rmdir(journal));
    }
    CHECK(!unlink(path)); free(f);
}
static void index_references(const char *root, unsigned n) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/reference.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/reference-journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    for (unsigned mode = 0; mode < 4; ++mode) {
        make(f, n, 0); prepared(f);
        aotx_ccir_input inputs[141];
        memcpy(inputs, f->input, f->count * sizeof(*inputs));
        aotx_ccir_input *extra = inputs + f->count;
        *extra = f->input[3]; extra->section.id[15] = 99; extra->section.flags = 0;
        if (mode != 2) {
            extra->section.schema = 99; extra->section.bytes = 8; extra->data = "BADINDEX";
        }
        if (mode == 1 || mode == 2) memcpy(f->manifest + 72, extra->section.id, 16);
        if (mode == 3) f->manifest[87] = 77;
        unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
        int made = aotx_ccir_create(path, lineage, inputs, f->count + 1, &meta, NULL);
        CHECK(mode ? made != 0 : made == 0);
        if (mode) CHECK(access(path, F_OK) != 0);
        if (!made) {
            aotx_runtime_boot boot;
            int rc = aotx_runtime_prepare(path, journal, 86, &boot);
            CHECK(mode ? rc != 0 : rc == 0);
            aotx_runtime_release(&boot);
            if (!rc) CHECK(!rmdir(journal));
            CHECK(!unlink(path));
        }
    }
    free(f);
}
int main(void) {
    char root[] = "/tmp/aotx-runtime-publication-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    test(root, 1, 0); test(root, 1, 1); test(root, 64, 0); test(root, 64, 1);
    compatibility(root, 1); compatibility(root, 64);
    shared_activation(root, 1); shared_activation(root, 64);
    shared_promotion(root, 1); shared_promotion(root, 64);
    checkpoint_ack(root, 1); checkpoint_ack(root, 64);
    index_references(root, 1); index_references(root, 64);
    CHECK(!rmdir(root));
    printf("runtime publication: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
