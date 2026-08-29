/* Purpose: Check that a module directory becomes one import of a head and complete parts.
 * Owns: One inbound ring and one temporary directory tree for each case.
 * Threading: One process; the check publishes into the ring and reads it back.
 * Lifetime: The run of the program. */
#include "disk/feed/import.h"
#include "tests/disk_fake.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

/* The ring holds every record of one case, so the check reads it after the publish and no
 * second thread is needed. The count is above the parts of a file of the bound. A build
 * whose bound guard is broken thus publishes and does not wait on the ring. */
#define AOTX_RING_SLOTS   8192u

/* The imports and the file bytes that one case may collect. */
#define AOTX_TEST_IMPORTS 70
#define AOTX_TEST_FILE    16384u

/* The bytes of the last part of a body, so a part count that rounds the wrong way fails. */
#define AOTX_TEST_TAIL    37u

/* 64 hexadecimal characters and one end byte. */
#define AOTX_TEST_HEX     65

static volatile sig_atomic_t stop_flag;

/* The import state holds two file buffers of one megabyte each, so it lives beside the
 * program and not on the stack. */
static aotx_import state;

typedef struct got_import {
    aotx_import_head head;
    uint32_t parts;                         /* parts that came after the head */
    uint32_t last_length;                   /* the byte count of the last part */
    uint32_t at[AOTX_IMPORT_FILES];         /* the offset that the next part must hold */
    uint32_t bytes[AOTX_IMPORT_FILES];      /* bytes assembled from the parts of each file */
    unsigned char text[AOTX_IMPORT_FILES][AOTX_TEST_FILE];
} got_import;

typedef struct collected {
    int count;                              /* heads that came */
    int records;                            /* records that came */
    got_import in[AOTX_TEST_IMPORTS];
} collected;

static collected got;

/* ---- the files of one module directory ---- */

static void make_dir(const char *path)
{
    CHECK(mkdir(path, 0755) == 0 || errno == EEXIST, "the directory %s does not open", path);
}

static void write_file(const char *path, const void *bytes, size_t len)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0, "the file %s does not open", path);
    if (fd >= 0) {
        CHECK(write(fd, bytes, len) == (ssize_t)len, "the file %s does not write", path);
        close(fd);
    }
}

static void write_text(const char *path, const char *text)
{
    write_file(path, text, strlen(text));
}

/* Gives the byte at one place of the body of one module. Each module gives another byte
 * run, so a part that carries the bytes of another file cannot pass. */
static unsigned char body_byte(int module, uint32_t at)
{
    return (unsigned char)(0x20u + ((uint32_t)module * 7u + at * 3u) % 90u);
}

/* Writes a body of the given byte count into the buffer. */
static void fill_body(unsigned char *out, int module, uint32_t len)
{
    uint32_t i;
    for (i = 0; i < len; i++) {
        out[i] = body_byte(module, i);
    }
}

/* ---- the ring, read the way a device consumer reads it ---- */

/* Gives the place of one import number in the table, or -1. */
static int place_of(collected *c, uint32_t number)
{
    int i;
    for (i = 0; i < c->count; i++) {
        if (c->in[i].head.import == number) {
            return i;
        }
    }
    return -1;
}

/* Takes every record the ring holds. A head opens an entry; a part must name an open
 * entry, hold the next part number, and start at the offset the file reached. */
static void take(aotx_inbound_ring *ring, collected *c)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        aotx_import_part part;
        int place;
        CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
        CHECK(h->type == AOTX_REC_IMPORT, "a slot holds the type %u", h->type);
        CHECK(h->cls == AOTX_CLASS_A, "an import record is not authoritative");
        CHECK(h->writer == AOTX_WRITER_FEEDER, "an import record holds the writer %u",
              h->writer);
        c->records++;
        memcpy(&part, aotx_record_body(h), sizeof(part));
        if (part.part == 0u) {
            CHECK(h->body_len == sizeof(aotx_import_head),
                  "a head has the body length %u", h->body_len);
            CHECK(c->count < AOTX_TEST_IMPORTS, "the case gives more imports than the table");
            if (c->count < AOTX_TEST_IMPORTS) {
                memset(&c->in[c->count], 0, sizeof(c->in[c->count]));
                memcpy(&c->in[c->count].head, aotx_record_body(h), sizeof(aotx_import_head));
                c->count++;
            }
        } else {
            CHECK(h->body_len == sizeof(aotx_import_part),
                  "a part has the body length %u", h->body_len);
            place = place_of(c, part.import);
            CHECK(place >= 0, "part %u names the import %u and no head opened it", part.part,
                  part.import);
            if (place >= 0) {
                got_import *g = &c->in[place];
                CHECK(part.part == g->parts + 1u, "part %u came after %u parts", part.part,
                      g->parts);
                CHECK(part.file < AOTX_IMPORT_FILES, "a part names the file %u", part.file);
                if (part.file < AOTX_IMPORT_FILES) {
                    CHECK(part.offset == g->at[part.file],
                          "a part of file %u starts at %u and %u was asked for", part.file,
                          part.offset, g->at[part.file]);
                    CHECK(part.length > 0 && part.length <= AOTX_IMPORT_TEXT_BYTES,
                          "a part carries %u bytes", part.length);
                    if (part.offset + part.length <= AOTX_TEST_FILE) {
                        memcpy(g->text[part.file] + part.offset, part.text, part.length);
                        g->bytes[part.file] = part.offset + part.length;
                    }
                    g->at[part.file] = part.offset + part.length;
                }
                g->parts++;
                g->last_length = part.length;
            }
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Opens a ring and an empty collection for one case. */
static void open_case(aotx_map *map, aotx_inbound_ring *ring)
{
    memset(&got, 0, sizeof(got));
    memset(&state, 0, sizeof(state));
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, map, ring) == 0, "the ring does not open");
}

/* Gives the parts that a file of the given byte count needs. */
static uint32_t parts_of(uint32_t len)
{
    return (len + AOTX_IMPORT_TEXT_BYTES - 1u) / AOTX_IMPORT_TEXT_BYTES;
}

/* Checks the head, the part count and the assembled bytes of one import. */
static void check_files(const got_import *g, int module, const char *name, uint32_t kind,
                        uint32_t manifest_bytes, uint32_t body_bytes)
{
    uint32_t want_parts = parts_of(manifest_bytes) + parts_of(body_bytes);
    uint32_t files = ((manifest_bytes > 0) ? 1u : 0u) + ((body_bytes > 0) ? 1u : 0u);
    uint32_t i;
    CHECK(strcmp(g->head.name, name) == 0, "the head names %s and %s was asked for",
          g->head.name, name);
    CHECK(g->head.kind == kind, "the head gives the kind %u and %u was asked for",
          g->head.kind, kind);
    CHECK(g->head.files == files, "the head counts %u files and %u carry bytes",
          g->head.files, files);
    CHECK(g->head.file_bytes[0] == manifest_bytes, "the head gives %u manifest bytes and %u"
          " were written", g->head.file_bytes[0], manifest_bytes);
    CHECK(g->head.file_bytes[1] == body_bytes, "the head gives %u body bytes and %u were"
          " written", g->head.file_bytes[1], body_bytes);
    CHECK(g->parts == want_parts, "the import gave %u parts and %u were asked for", g->parts,
          want_parts);
    CHECK(g->bytes[1] == body_bytes, "the parts of the body assembled %u bytes and the file"
          " holds %u", g->bytes[1], body_bytes);
    if (body_bytes > 0 && body_bytes <= AOTX_TEST_FILE) {
        uint32_t bad = body_bytes;
        for (i = 0; i < body_bytes; i++) {
            if (g->text[1][i] != body_byte(module, i)) {
                bad = i;
                break;
            }
        }
        CHECK(bad == body_bytes, "byte %u of the assembled body is 0x%02x and 0x%02x was"
              " written", bad, g->text[1][bad], body_byte(module, bad));
    }
}

/* ---- the three kinds of module ---- */

/* Gives the digest that the system tool computes. The tool is the outside reference for
 * the value that the head must carry. */
static void tool_digest(const char *path, char *text)
{
    char command[2048];
    FILE *pipe;
    text[0] = '\0';
    snprintf(command, sizeof(command), "sha256sum '%s' 2>/dev/null", path);
    pipe = popen(command, "r");
    if (pipe == NULL) {
        return;
    }
    if (fscanf(pipe, "%64s", text) != 1) {
        text[0] = '\0';
    }
    pclose(pipe);
}

/* Three kinds of module: a skill directory of the open shape, a role directory with a
 * manifest and an overlay, and a device tool directory. The head of the tool must carry
 * the digest of the module file. */
static void kinds(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    unsigned char body[AOTX_TEST_FILE];
    char dir[256];
    char path[512];
    char file[768];
    char manifest[512];
    char role_text[512];
    char want[AOTX_TEST_HEX];
    char text[AOTX_TEST_HEX];
    const char *reason = "";
    uint32_t skill_bytes = 700u;
    uint32_t overlay_bytes = 400u;
    uint32_t role_manifest;
    uint32_t tool_manifest;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    open_case(&map, &ring);

    /* A skill directory holds a skill file and no manifest. */
    snprintf(path, sizeof(path), "%s/a_skill", dir);
    make_dir(path);
    fill_body(body, 1, skill_bytes);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
    write_file(file, body, skill_bytes);
    CHECK(aotx_import_dir(&state, path, &ring, &stop_flag, &reason) == 0,
          "the skill directory is refused: %s", reason);

    /* A role directory holds a manifest and the overlay that the body key names. */
    snprintf(path, sizeof(path), "%s/a_role", dir);
    make_dir(path);
    role_manifest = (uint32_t)snprintf(role_text, sizeof(role_text),
                                       "# the role of a run\nkind: role\nname: a_role\n"
                                       "model: language\nbody: overlay.txt\n");
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_file(file, role_text, role_manifest);
    fill_body(body, 2, overlay_bytes);
    snprintf(file, sizeof(file), "%s/overlay.txt", path);
    write_file(file, body, overlay_bytes);
    CHECK(aotx_import_dir(&state, path, &ring, &stop_flag, &reason) == 0,
          "the role directory is refused: %s", reason);

    /* A tool directory of a device tool holds a manifest and a module file. The module
     * file is not published: the head carries its digest. */
    snprintf(path, sizeof(path), "%s/a_tool", dir);
    make_dir(path);
    tool_manifest = (uint32_t)snprintf(manifest, sizeof(manifest),
                                       "kind: tool\nname: a_tool\nside: device\n"
                                       "module: tool.ptx\nentry: aotx_tool_a_tool\n");
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_file(file, manifest, tool_manifest);
    fill_body(body, 3, 2000u);
    snprintf(file, sizeof(file), "%s/tool.ptx", path);
    write_file(file, body, 2000u);
    CHECK(aotx_import_dir(&state, path, &ring, &stop_flag, &reason) == 0,
          "the tool directory is refused: %s", reason);

    take(&ring, &got);
    CHECK(got.count == 3, "the three directories gave %d imports", got.count);
    if (got.count == 3) {
        check_files(&got.in[0], 1, "a_skill", AOTX_MODULE_SKILL, 0u, skill_bytes);
        check_files(&got.in[1], 2, "a_role", AOTX_MODULE_ROLE, role_manifest, overlay_bytes);
        check_files(&got.in[2], 3, "a_tool", AOTX_MODULE_TOOL, tool_manifest, 0u);
        for (i = 0; i < 3; i++) {
            CHECK(got.in[i].head.import == (uint32_t)(i + 1),
                  "import %d holds the number %u", i, got.in[i].head.import);
        }
        /* The manifest of the role and of the tool assembles from its parts. */
        CHECK(got.in[1].bytes[0] == role_manifest &&
              memcmp(got.in[1].text[0], role_text, role_manifest) == 0,
              "the manifest of the role does not assemble to the file");
        CHECK(got.in[2].bytes[0] == tool_manifest &&
              memcmp(got.in[2].text[0], manifest, tool_manifest) == 0,
              "the manifest of the tool does not assemble to the file");
        /* The digest of the module file is the one the system tool computes. */
        snprintf(file, sizeof(file), "%s/a_tool/tool.ptx", dir);
        tool_digest(file, want);
        aotx_sha256_text(got.in[2].head.digest, text);
        CHECK(want[0] == '\0' || strcmp(text, want) == 0,
              "the head gives the digest %s and the tool gives %s", text, want);
        /* The other two kinds carry no digest. */
        for (i = 0; i < 2; i++) {
            aotx_sha256_text(got.in[i].head.digest, text);
            CHECK(strcmp(text, "00000000000000000000000000000000"
                               "00000000000000000000000000000000") == 0,
                  "import %d carries a digest and the kind names no module file", i);
        }
        /* The head names the tail of the path, so the console can show where it came from. */
        CHECK(strstr(got.in[0].head.path, "a_skill") != NULL,
              "the head gives the path %s", got.in[0].head.path);
    }
    printf("kinds: imports %d, records %d\n", got.count, got.records);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* ---- a walk of one directory of module directories ---- */

/* Builds n module directories with distinct names and distinct bodies. Imports the whole
 * directory. Checks the name order, the import numbers and the assembled bytes. */
static void tree(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    unsigned char body[AOTX_TEST_FILE];
    char dir[256];
    char root[320];
    char path[512];
    char file[768];
    char manifest[512];
    char name[32];
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    open_case(&map, &ring);
    snprintf(root, sizeof(root), "%s/modules", dir);
    make_dir(root);
    for (i = 0; i < n; i++) {
        uint32_t len = 100u + (uint32_t)i * 13u;
        snprintf(name, sizeof(name), "m%02d", i);
        snprintf(path, sizeof(path), "%s/%s", root, name);
        make_dir(path);
        snprintf(manifest, sizeof(manifest), "kind: skill\nname: %s\nbody: skill.txt\n", name);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        write_text(file, manifest);
        fill_body(body, i, len);
        snprintf(file, sizeof(file), "%s/skill.txt", path);
        write_file(file, body, len);
    }
    /* A name that starts with a dot is not a module directory and makes no import. */
    snprintf(path, sizeof(path), "%s/.hidden", root);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "kind: skill\nname: hidden\n");

    CHECK(aotx_import_tree(&state, root, &ring, &stop_flag) == 0, "the walk does not finish");
    take(&ring, &got);
    CHECK(got.count == n, "the walk gave %d imports and %d directories were written",
          got.count, n);
    CHECK((int)state.imports == n, "the state counts %llu imports",
          (unsigned long long)state.imports);
    for (i = 0; i < n && i < got.count; i++) {
        uint32_t len = 100u + (uint32_t)i * 13u;
        uint32_t manifest_bytes;
        snprintf(name, sizeof(name), "m%02d", i);
        snprintf(manifest, sizeof(manifest), "kind: skill\nname: %s\nbody: skill.txt\n", name);
        manifest_bytes = (uint32_t)strlen(manifest);
        /* The walk takes the directories in name order, so place i holds name i. */
        check_files(&got.in[i], i, name, AOTX_MODULE_SKILL, manifest_bytes, len);
        CHECK(got.in[i].head.import == (uint32_t)(i + 1),
              "the import at place %d holds the number %u", i, got.in[i].head.import);
        CHECK(got.in[i].bytes[0] == manifest_bytes &&
              memcmp(got.in[i].text[0], manifest, manifest_bytes) == 0,
              "the manifest of module %d does not assemble to the file", i);
    }
    printf("tree %d: imports %d, records %d\n", n, got.count, got.records);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* ---- a body of several parts ---- */

/* A body of k parts, with a short last part. A part count that rounds the wrong way then
 * cannot pass, and a last part that carries a whole part cannot pass. */
static void parts(int k)
{
    aotx_map map;
    aotx_inbound_ring ring;
    static unsigned char body[AOTX_TEST_FILE];
    char dir[256];
    char path[512];
    char file[768];
    const char *reason = "";
    uint32_t len = (uint32_t)(k - 1) * AOTX_IMPORT_TEXT_BYTES + AOTX_TEST_TAIL;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    open_case(&map, &ring);
    snprintf(path, sizeof(path), "%s/long_body", dir);
    make_dir(path);
    fill_body(body, 9, len);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
    write_file(file, body, len);
    CHECK(aotx_import_dir(&state, path, &ring, &stop_flag, &reason) == 0,
          "the directory is refused: %s", reason);
    take(&ring, &got);
    CHECK(got.count == 1, "the directory gave %d imports", got.count);
    if (got.count == 1) {
        check_files(&got.in[0], 9, "long_body", AOTX_MODULE_SKILL, 0u, len);
        CHECK(got.in[0].parts == (uint32_t)k, "the body gave %u parts and %d were asked for",
              got.in[0].parts, k);
        CHECK(got.in[0].last_length == AOTX_TEST_TAIL,
              "the last part carries %u bytes and %u were asked for", got.in[0].last_length,
              (unsigned)AOTX_TEST_TAIL);
        CHECK(got.records == k + 1, "the import gave %d records and %d were asked for",
              got.records, k + 1);
    }
    printf("parts %d: parts %u, records %d\n", k, got.in[0].parts, got.records);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* ---- every refusal, each with the reason it states ---- */

/* Imports one directory that must be refused, and checks the reason and that no record
 * reached the ring. */
static void refuse(aotx_inbound_ring *ring, const char *path, const char *want)
{
    const char *reason = "";
    uint64_t before = aotx_inbound_head(ring);
    uint64_t refusals = state.refusals;
    int status = aotx_import_dir(&state, path, ring, &stop_flag, &reason);
    CHECK(status == 1, "the directory %s gives the status %d and a refusal was asked for",
          path, status);
    CHECK(strstr(reason, want) != NULL, "the refusal of %s states [%s] and [%s] was asked"
          " for", path, reason, want);
    CHECK(aotx_inbound_head(ring) == before, "a refused directory published a record");
    CHECK(state.refusals == refusals + 1u, "a refused directory is not counted");
}

static void refusals(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    static unsigned char body[AOTX_IMPORT_CAP + 1u];
    char dir[256];
    char path[512];
    char link[512];
    char file[768];

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    open_case(&map, &ring);

    /* A directory with neither file names no module. */
    snprintf(path, sizeof(path), "%s/no_files", dir);
    make_dir(path);
    refuse(&ring, path, "no manifest and no skill file");

    /* A name outside the rule of the catalog. */
    snprintf(path, sizeof(path), "%s/Bad-Name", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
    write_text(file, "a skill body\n");
    refuse(&ring, path, "lower case letters, digits and underscores");

    /* A name of more than sixty-three bytes. */
    snprintf(path, sizeof(path), "%s/%s", dir,
             "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffgggg");
    make_dir(path);
    refuse(&ring, path, "lower case letters, digits and underscores");

    /* A file over the bound of the feeder. */
    snprintf(path, sizeof(path), "%s/too_big", dir);
    make_dir(path);
    memset(body, 'x', sizeof(body));
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
    write_file(file, body, sizeof(body));
    refuse(&ring, path, "longer than the bound");

    /* A symbolic link at the last component of the path. */
    snprintf(path, sizeof(path), "%s/a_module", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
    write_text(file, "a skill body\n");
    snprintf(link, sizeof(link), "%s/linked", dir);
    CHECK(symlink(path, link) == 0, "the link does not open");
    refuse(&ring, link, "symbolic link");

    /* A symbolic link at a component that is not the last one. */
    snprintf(file, sizeof(file), "%s/a_module", link);
    make_dir(file);
    refuse(&ring, file, "symbolic link");

    /* A skill file that is a symbolic link. */
    snprintf(path, sizeof(path), "%s/linked_body", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
    snprintf(link, sizeof(link), "%s/a_module/%s", dir, AOTX_IMPORT_SKILL);
    CHECK(symlink(link, file) == 0, "the link does not open");
    refuse(&ring, path, "symbolic link");

    /* A manifest that is a symbolic link names its own defect, and does not read as a
     * directory that holds no manifest. */
    snprintf(path, sizeof(path), "%s/linked_manifest", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    snprintf(link, sizeof(link), "%s/a_module/%s", dir, AOTX_IMPORT_SKILL);
    CHECK(symlink(link, file) == 0, "the link does not open");
    refuse(&ring, path, "symbolic link");

    /* A manifest that names a body file that is not there. */
    snprintf(path, sizeof(path), "%s/no_body", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "kind: role\nname: no_body\nbody: absent.txt\n");
    refuse(&ring, path, "the file is not there");

    /* A manifest with no kind, and a manifest with a kind that names no module kind. */
    snprintf(path, sizeof(path), "%s/no_kind", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "name: no_kind\ndescription: a module with no kind\n");
    refuse(&ring, path, "names no kind");

    snprintf(path, sizeof(path), "%s/bad_kind", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "kind: agent\nname: bad_kind\n");
    refuse(&ring, path, "not a skill, a role or a tool");

    /* A device tool that names no module file, and one whose module file is not there. */
    snprintf(path, sizeof(path), "%s/no_module", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "kind: tool\nname: no_module\nside: device\n");
    refuse(&ring, path, "names no module file");

    snprintf(path, sizeof(path), "%s/lost_module", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "kind: tool\nname: lost_module\nside: device\nmodule: gone.ptx\n");
    refuse(&ring, path, "the file is not there");

    /* A body file that leaves the module directory. */
    snprintf(path, sizeof(path), "%s/up_body", dir);
    make_dir(path);
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    write_text(file, "kind: role\nname: up_body\nbody: ../a_module/SKILL.md\n");
    refuse(&ring, path, "two dots");

    /* A directory that is not there. */
    snprintf(path, sizeof(path), "%s/absent", dir);
    refuse(&ring, path, "the file is not there");

    take(&ring, &got);
    CHECK(got.count == 0 && got.records == 0, "a refused case published %d records",
          got.records);
    printf("refusals: cases %llu, records %d\n", (unsigned long long)state.refusals,
           got.records);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* ---- the line that asks for an import ---- */

/* Each line is taken or left. A line that the feeder takes gives the path it names. */
static void lines(void)
{
    static const struct {
        const char *line;
        int taken;
        const char *path;
    } cases[10] = {
        { "import modules/a_skill", 1, "modules/a_skill" },
        { "import   /var/lib/a_role  ", 1, "/var/lib/a_role" },
        { "import\tmodules/a_tool", 1, "modules/a_tool" },
        { "import", 0, "" },
        { "import ", 0, "" },
        { "import   ", 0, "" },
        { "imports modules/a", 0, "" },
        { "importmodules", 0, "" },
        { " import modules/a", 0, "" },
        { "spawn worker", 0, "" }
    };
    char path[AOTX_WALK_BYTES];
    int i;
    for (i = 0; i < 10; i++) {
        int taken = aotx_import_line((const unsigned char *)cases[i].line,
                                     (uint32_t)strlen(cases[i].line), path, sizeof(path));
        CHECK(taken == cases[i].taken, "the line [%s] gives %d and %d was asked for",
              cases[i].line, taken, cases[i].taken);
        if (taken && cases[i].taken) {
            CHECK(strcmp(path, cases[i].path) == 0, "the line [%s] gives the path [%s]",
                  cases[i].line, path);
        }
    }
    printf("lines: cases %d\n", 10);
}

int main(void)
{
    kinds();
    tree(1);
    tree(64);
    parts(3);
    parts(64);
    refusals();
    lines();
    return aotx_report("import_test", 700);
}
