/* Purpose: Give the feeder check the arm of the line that states a refused import.
 * Owns: Nothing; the check that includes this file holds the ring and the child process.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: One case of the check.
 *
 * The file is included by tests/feed_test.c after tests/feed_import.h, whose parts it
 * uses. */
#ifndef AOTX_TESTS_FEED_REFUSE_H
#define AOTX_TESTS_FEED_REFUSE_H

/* The causes that make the feeder refuse an import line. The batch takes them in turn, so
 * every cause comes at N=64 and one of them at N=1. */
#define AOTX_REFUSE_CAUSES 10

/* The bytes of the line that one refusal must give. The buffer is wider than a record
 * body. A case that gives a longer line thus fails the comparison, and does not match a
 * line that the feeder cut. */
#define AOTX_REFUSE_LINE 1024

/* One component of a path that is long enough to cut the line of a refusal. */
#define AOTX_REFUSE_DEEP "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffff"

static char refuse_want[AOTX_LINES_MAX][AOTX_REFUSE_LINE];

/* Builds the directory of one refused import under the root. Writes the path that the
 * import line names and the reason that the feeder must state for it. */
static void refuse_case(const char *root, int i, char *path, size_t path_bytes,
                        char *reason, size_t reason_bytes)
{
    static unsigned char big[AOTX_IMPORT_CAP + 1u];
    char file[1024];
    char target[1024];
    int cause = i % AOTX_REFUSE_CAUSES;
    if (cause == 0) {
        /* A name outside the rule of the catalog. */
        snprintf(path, path_bytes, "%s/Bad-Name-%d", root, i);
        module_dir(path);
        snprintf(reason, reason_bytes, "the name is not one to sixty-three bytes of lower"
                                       " case letters, digits and underscores");
    } else if (cause == 1) {
        snprintf(path, path_bytes, "%s/empty_%d", root, i);
        module_dir(path);
        snprintf(reason, reason_bytes, "the directory holds no manifest and no skill file");
    } else if (cause == 2) {
        /* A symbolic link at the last component of the path. */
        snprintf(target, sizeof(target), "%s/target_%d", root, i);
        module_dir(target);
        snprintf(file, sizeof(file), "%.900s/%s", target, AOTX_IMPORT_SKILL);
        module_text(file, "a skill body\n");
        snprintf(path, path_bytes, "%s/linked_%d", root, i);
        CHECK(symlink(target, path) == 0, "the link does not open");
        snprintf(reason, reason_bytes, "a component of the path is a symbolic link");
    } else if (cause == 3) {
        /* A file over the bound of the feeder. */
        snprintf(path, path_bytes, "%s/big_%d", root, i);
        module_dir(path);
        memset(big, 'x', sizeof(big));
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_SKILL);
        module_file(file, big, sizeof(big));
        snprintf(reason, reason_bytes, "the skill file is longer than the bound of %u bytes",
                 (unsigned)AOTX_IMPORT_CAP);
    } else if (cause == 4) {
        snprintf(path, path_bytes, "%s/no_body_%d", root, i);
        module_dir(path);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_text(file, "kind: role\nname: a_role\nbody: absent.txt\n");
        snprintf(reason, reason_bytes, "the body file is not there");
    } else if (cause == 5) {
        snprintf(path, path_bytes, "%s/no_file_%d", root, i);
        module_dir(path);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_text(file, "kind: tool\nname: a_tool\nside: device\nmodule: gone.ptx\n");
        snprintf(reason, reason_bytes, "the module file is not there");
    } else if (cause == 6) {
        /* A file of the manifest that leaves the module directory. */
        snprintf(path, path_bytes, "%s/up_body_%d", root, i);
        module_dir(path);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_text(file, "kind: role\nname: a_role\nbody: ../m00/skill.txt\n");
        snprintf(reason, reason_bytes, "the body file: the path holds a component of two"
                                       " dots");
    } else if (cause == 7) {
        snprintf(path, path_bytes, "%s/no_kind_%d", root, i);
        module_dir(path);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_text(file, "name: a_module\ndescription: a module with no kind\n");
        snprintf(reason, reason_bytes, "the manifest names no kind");
    } else if (cause == 8) {
        snprintf(path, path_bytes, "%s/bad_kind_%d", root, i);
        module_dir(path);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_text(file, "kind: agent\nname: a_module\n");
        snprintf(reason, reason_bytes, "the manifest names a kind that is not a skill, a"
                                       " role or a tool");
    } else {
        snprintf(path, path_bytes, "%s/no_module_%d", root, i);
        module_dir(path);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_text(file, "kind: tool\nname: a_tool\nside: device\n");
        snprintf(reason, reason_bytes, "the manifest of a device tool names no module file");
    }
}

/* An import line that the feeder refuses gives one input line record of the shape
 * "import <path> refused: <reason>". The device parser prints that line, so the operator
 * who typed the line sees the cause. A line that the feeder takes gives no such record,
 * because the catalog states the commit. */
static void refused_line_arm(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    import_taken got;
    char dir[256];
    char root[320];
    char err_path[400];
    char report[16384];
    char path[512];
    char deep[1024];
    char reason[256];
    char line[1024];
    char fd_text[16];
    char *args[4];
    int pipe_fds[2];
    int err_fd;
    int saved;
    int child;
    int i;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/modules", dir);
    /* One module directory that imports, so the arm proves that a line the feeder takes
     * gives no refusal line. It also gives the file that the two dots case reaches for. */
    build_modules(root, 1);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);

    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = NULL;
    err_fd = open(err_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    saved = dup(2);
    CHECK(err_fd >= 0 && saved >= 0, "the report file does not open");
    fflush(stderr);
    dup2(err_fd, 2);
    child = aotx_spawn(args, pipe_fds[0], -1);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    close(err_fd);
    CHECK(child > 0, "the feeder does not start");
    close(pipe_fds[0]);

    for (i = 0; i < n; i++) {
        int used;
        refuse_case(root, i, path, sizeof(path), reason, sizeof(reason));
        snprintf(refuse_want[i], AOTX_REFUSE_LINE, "import %s refused: %s", path, reason);
        used = snprintf(line, sizeof(line), "import %s\n", path);
        CHECK(write(pipe_fds[1], line, (size_t)used) == used,
              "the import line does not write");
    }
    /* A path that makes the line longer than a body gives the tail of the path and the
     * whole reason. The reason is what the operator does not know. */
    {
        int used;
        snprintf(path, sizeof(path), "%.400s/%s", root, AOTX_REFUSE_DEEP);
        module_dir(path);
        snprintf(line, sizeof(line), "%.400s/%s", path, AOTX_REFUSE_DEEP);
        module_dir(line);
        snprintf(deep, sizeof(deep), "%.900s/deep_module", line);
        module_dir(deep);
        used = snprintf(line, sizeof(line), "import %s\n", deep);
        CHECK(strlen(deep) > AOTX_BODY_BYTES / 2, "the deep path is not long enough");
        CHECK(write(pipe_fds[1], line, (size_t)used) == used, "the line does not write");
    }
    /* One line that the feeder takes. It must give the import and no refusal line. */
    {
        int used = snprintf(line, sizeof(line), "import %s/m00\n", root);
        CHECK(write(pipe_fds[1], line, (size_t)used) == used, "the line does not write");
    }

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while ((got.lines < n + 1 || got.heads < 1) && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        take_imports(&ring, &got);
        aotx_pause(&backoff);
    }
    close(pipe_fds[1]);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    take_imports(&ring, &got);

    CHECK(got.lines == n + 1, "the run gave %d refusal lines and %d were asked for",
          got.lines, n + 1);
    CHECK(got.heads == 1, "the run gave %d imports and one was asked for", got.heads);
    for (i = 0; i < n && i < got.lines; i++) {
        CHECK(strcmp(got.text[i], refuse_want[i]) == 0,
              "refusal line %d holds [%s] and [%s] was asked for", i, got.text[i],
              refuse_want[i]);
    }
    /* The cut line keeps the whole reason and the tail of the path. */
    if (got.lines == n + 1) {
        const char *cut = got.text[n];
        size_t bytes = strlen(cut);
        snprintf(reason, sizeof(reason),
                 " refused: the directory holds no manifest and no skill file");
        CHECK(bytes == AOTX_BODY_BYTES, "the cut line holds %u bytes and %u were asked for",
              (unsigned)bytes, (unsigned)AOTX_BODY_BYTES);
        CHECK(strncmp(cut, "import ", 7) == 0, "the cut line does not start with the word");
        CHECK(bytes > strlen(reason) &&
              strcmp(cut + bytes - strlen(reason), reason) == 0,
              "the cut line does not end with the whole reason: [%s]", cut);
        if (bytes > strlen(reason) + 7u) {
            size_t take = bytes - strlen(reason) - 7u;
            CHECK(strlen(deep) > take && strncmp(deep + strlen(deep) - take, cut + 7,
                                                 take) == 0,
                  "the cut line does not hold the tail of the path: [%s]", cut);
        }
    }
    /* The line of the standard error stays beside the record. */
    read_report(err_path, report, sizeof(report));
    CHECK(import_count(report, "import: ") == n + 1, "the standard error holds %d refusal"
          " lines and %d were asked for", import_count(report, "import: "), n + 1);
    printf("refused lines %d: refusal lines %d, imports %d\n", n, got.lines, got.heads);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

#endif
