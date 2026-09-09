/* Purpose: Check typed GPU state through the real CCIR file and command consumers.
 * Owns: Temporary container files, exact byte oracles and process results.
 * Launch shape: Distinct N=1 and N=64 object, media and appraisal batches.
 * Lifetime: One test process and bounded child exports. */
#include "cognitive_fixture.h"
#include "cognitive/io.h"
#include <sys/wait.h>
#include <unistd.h>

static int aotx_run(const char *program, const char *input, const char *output) {
    pid_t child = fork();
    if (child == 0) { execl(program, program, input, output, (char *)NULL); _exit(127); }
    if (child < 0) return -1;
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}
static aotx_fixture aotx_file_media(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_MEDIA, 501 + i, n + i + 1);
        aotx_bytes p(195, 0);
        aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, AOTX_COG_IMAGE_MEDIA, 4);
        aotx_put(p.data() + 8, AOTX_COG_SOURCE_BYTES, 4); aotx_put(p.data() + 12, AOTX_COG_U8, 4);
        aotx_put(p.data() + 16, 1); aotx_put(p.data() + 24, 1); aotx_put(p.data() + 32, 3);
        aotx_put(p.data() + 48, 3); aotx_put(p.data() + 64, AOTX_COG_SPATIAL, 4);
        aotx_put(p.data() + 68, 3, 4); p[72] = 1;
        p[192] = (unsigned char)i; p[193] = (unsigned char)(127 + i); p[194] = (unsigned char)(255 - i);
        f.add(r, p);
    }
    return f;
}
static void aotx_file_case(unsigned n, const char *program, const char *directory) {
    std::string input = std::string(directory) + "/input-" + std::to_string(n) + ".aotxccir";
    std::string output = std::string(directory) + "/output-" + std::to_string(n) + ".aotxccir";
    std::string next = std::string(directory) + "/next-" + std::to_string(n) + ".aotxccir";
    aotx_bytes expected;
    {
        auto base = aotx_initial(n), media = aotx_file_media(n), tail = aotx_appraisals(n);
        base.append(media);
        for (unsigned i = 0; i < n; ++i) {
            aotx_put(tail.rows[i].data() + AOTX_CO_CREATED, 2 * n + i + 1);
            aotx_put(tail.rows[i].data() + AOTX_CO_UPDATED, 2 * n + i + 1);
        }
        aotx_device d;
        aotx_check(!d.load(base.wire(false, 2 * n)).status, "file checkpoint GPU admission");
        auto checkpoint = d.checkpoint(), log = tail.wire(true, 2 * n + 1, 6);
        unsigned char lineage[16], ids[4][16], manifest[96];
        aotx_id(lineage, 9000);
        for (unsigned i = 0; i < 4; ++i) aotx_id(ids[i], 6000 + i);
        aotx_ccir_manifest(manifest, ids[1], ids[2]);
        unsigned char optional[7] = {0, 17, 255, (unsigned char)n, 4, 5, 6};
        aotx_ccir_input sections[4] = {};
        for (unsigned i = 0; i < 4; ++i) {
            sections[i].section.type = i < 3 ? i + 1 : 8000;
            sections[i].section.schema = i < 3 ? 1 : 9;
            sections[i].section.flags = i < 3 ? AOTX_CCIR_REQUIRED : 0;
            sections[i].section.alignment = 64;
            memcpy(sections[i].section.id, ids[i], 16);
        }
        sections[0].data = manifest; sections[0].section.bytes = sizeof(manifest);
        sections[1].data = checkpoint.data(); sections[1].section.bytes = checkpoint.size();
        sections[2].data = log.data(); sections[2].section.bytes = log.size();
        sections[3].data = optional; sections[3].section.bytes = sizeof(optional);
        aotx_ccir_meta meta = {2 * n, 3 * n, 6};
        aotx_check(aotx_ccir_create(input.c_str(), lineage, sections, 4, &meta, NULL) == 0, "create GPU state CCIR");
        aotx_check(!d.load(log, true).status, "original GPU tail admission");
        base.append(tail); expected = base.wire(false, 3 * n, 6);
        aotx_check(d.checkpoint() == expected, "independent full state oracle");
        std::fill(checkpoint.begin(), checkpoint.end(), 0); std::fill(log.begin(), log.end(), 0);
    }
    /* All original device and source buffers have been released before the child starts. */
    aotx_check(aotx_run(program, input.c_str(), output.c_str()) == 0, "actual GPU export command");
    aotx_cognitive_file file;
    int status = aotx_cognitive_file_open(output.c_str(), &file);
    aotx_check(!status, "open exported state file");
    if (status) return;
    aotx_check(file.view.meta.checkpoint_sequence == 3 * n && file.view.meta.durable_sequence == 3 * n &&
               !file.tail_bytes, "tail folded into committed checkpoint");
    aotx_bytes restored(file.checkpoint, file.checkpoint + file.checkpoint_bytes);
    aotx_check(restored == expected, "file round trip exact objects and media");
    aotx_check(file.view.count == 3, "optional extent retained");
    unsigned char optional[7] = {}, wanted[7] = {0, 17, 255, (unsigned char)n, 4, 5, 6};
    unsigned optional_hits = 0;
    for (uint32_t i = 0; i < file.view.count; ++i) if (file.view.sections[i].type == 8000) {
        ++optional_hits;
        aotx_ccir_read read = {i, 0, sizeof(optional), optional};
        aotx_check(!aotx_ccir_read_batch(&file.view, &read, 1) && !memcmp(optional, wanted, sizeof(wanted)),
                   "unknown optional bytes preserved");
    }
    aotx_check(optional_hits == 1, "optional extent comparison executed once");
    aotx_cognitive_file_close(&file);
    aotx_device fresh;
    aotx_check(!fresh.load(restored).status && fresh.checkpoint() == expected, "fresh device restore from file");
    for (const auto &m : fresh.resolve(n, 101, 1)) aotx_check(!m.status, "restored appraisal access");
    auto selections = aotx_selections(n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_put(selections.rows[i].data() + AOTX_CO_CREATED, 3 * n + i + 1);
        aotx_put(selections.rows[i].data() + AOTX_CO_UPDATED, 3 * n + i + 1);
        aotx_id(selections.payloads[i].data() + 48, 501 + i);
        aotx_put(selections.payloads[i].data() + 72, 2, 4);
    }
    auto applied = fresh.load(selections.wire(true, 3 * n + 1, 7), true);
    aotx_check(!applied.status && applied.sequence == 4 * n && applied.applied == n, "new mutation after file restore");
    for (const auto &m : fresh.resolve(n, 201, 1)) aotx_check(!m.status, "restored text and media selection");
    aotx_check(aotx_run(program, output.c_str(), next.c_str()) == 0, "checkpoint-only file command");
    aotx_check(aotx_run(program, input.c_str(), output.c_str()) == 1, "command refuses existing output");
    aotx_check(!aotx_cognitive_file_open(next.c_str(), &file), "open second export");
    if (file.view.fd >= 0) {
        aotx_check(file.checkpoint_bytes == expected.size() && !memcmp(file.checkpoint, expected.data(), expected.size()),
                   "second export exact bytes");
        aotx_cognitive_file_close(&file);
    }
    unlink(input.c_str()); unlink(output.c_str()); unlink(next.c_str());
    printf("cognitive file N=%u complete\n", n);
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char directory[] = "/tmp/aotx-cognitive-XXXXXX";
    if (!mkdtemp(directory)) return 2;
    for (unsigned n : {1u, 64u}) aotx_file_case(n, argv[1], directory);
    rmdir(directory);
    printf("cognitive file: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
