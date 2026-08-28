/* Purpose: Check the text that the journal reader prints for the token records of a run.
 * Owns: One temporary journal for each case.
 * Threading: Two processes; the test reads the file that the reader writes.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>

#define AOTX_LINES_MAX 4096
#define AOTX_TEXT_MAX  262144

static char **arguments;
static char text[AOTX_TEXT_MAX];
static char *lines[AOTX_LINES_MAX];

/* Marks the record that went in last as one that a restore applied again. */
static void mark_replayed(aotx_fake_device *d)
{
    aotx_record_header *h = (aotx_record_header *)(d->stage + AOTX_BLOCK_HEADER_BYTES +
                                                   (size_t)(d->count - 1) * AOTX_SLOT_BYTES);
    h->flags = (uint16_t)(h->flags | AOTX_FLAG_REPLAYED);
}

/* Writes one boot directory with n ticks. Each tick holds one token record. The last tick
 * adds a token record whose body is too short, which the reader must not print. */
static void build_journal(const char *dir, uint64_t boot_id, int n, int replayed)
{
    aotx_fake_device device;
    aotx_segment_writer writer;
    char boot_dir[320];
    char body[64];
    int i;
    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    CHECK(aotx_make_dir(boot_dir) == 0, "the boot directory does not open");
    CHECK(aotx_segment_open(&writer, boot_dir, 4096) == 0, "the segment does not open");
    aotx_fake_start(&device, NULL, boot_id);
    for (i = 0; i < n; i++) {
        aotx_boot_body boot;
        aotx_clock_body clock;
        aotx_commit_body commit;
        aotx_token_body token;
        const aotx_block_header *h;
        if (i == 0) {
            memset(&boot, 0, sizeof(boot));
            boot.boot_id = boot_id;
            boot.wall_ns = 1700000000000000000ull + boot_id;
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_BOOT, &boot, sizeof(boot));
        }
        clock.wall_ns = 1700000000000000000ull + (uint64_t)i;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
        snprintf(body, sizeof(body), "line %d", i);
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_INPUT_LINE, body, (uint32_t)strlen(body));
        aotx_fake_token(i, &token);
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, sizeof(token));
        if (replayed) {
            mark_replayed(&device);
        }
        if (i == n - 1) {
            /* A body that is shorter than the layout holds no token, so the reader counts
             * it and prints nothing for it. */
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, 8u);
        }
        memset(&commit, 0, sizeof(commit));
        commit.state_hash = 0x00bb000000000000ull + (uint64_t)i + 1;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
        aotx_fake_commit(&device, 0);
        h = (const aotx_block_header *)device.stage;
        CHECK(aotx_segment_put(&writer, device.stage, h->byte_len) == 0, "a frame does not write");
    }
    CHECK(aotx_segment_close(&writer) == 0, "the segment does not close");
}

/* Runs the reader and returns its exit status. The lines go to the file that out_path
 * names, and the reader writes its own report to the standard error. */
static int run_reader(const char *dir, const char *boot, const char *out_path)
{
    char *args[6];
    int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    int child;
    CHECK(out_fd >= 0, "the output file does not open");
    args[0] = arguments[1];
    args[1] = (char *)"tokens";
    args[2] = (char *)dir;
    if (boot != NULL) {
        args[3] = (char *)"--boot";
        args[4] = (char *)boot;
        args[5] = NULL;
    } else {
        args[3] = NULL;
    }
    child = aotx_spawn(args, -1, out_fd);
    CHECK(child > 0, "the reader does not start");
    close(out_fd);
    return aotx_wait(child);
}

/* Reads the file and gives the count of lines. Each line loses its end byte. */
static int split(const char *path)
{
    int fd = open(path, O_RDONLY);
    ssize_t got;
    int count = 0;
    char *at;
    CHECK(fd >= 0, "the output file does not read");
    if (fd < 0) {
        return 0;
    }
    got = read(fd, text, sizeof(text) - 1);
    close(fd);
    CHECK(got >= 0, "the output file gives no bytes");
    if (got < 0) {
        return 0;
    }
    text[got] = '\0';
    at = text;
    while (*at != '\0' && count < AOTX_LINES_MAX) {
        char *end = strchr(at, '\n');
        lines[count++] = at;
        if (end == NULL) {
            break;
        }
        *end = '\0';
        at = end + 1;
    }
    return count;
}

/* Checks that each line names the token that the number gives, in order. */
static void compare(int count, int n, int replayed)
{
    int i;
    CHECK(count == n, "the reader printed %d lines and %d tokens are in the journal", count, n);
    for (i = 0; i < count && i < n; i++) {
        aotx_token_body token;
        char want[256];
        char tail[32];
        aotx_fake_token(i, &token);
        snprintf(want, sizeof(want),
                 "slot=%u position=%u token=%u flags=0x%04x seed=%016llx draw=%llu role=%u",
                 token.slot, token.position, token.token, token.flags,
                 (unsigned long long)token.seed, (unsigned long long)token.draw, token.role);
        CHECK(strncmp(lines[i], want, strlen(want)) == 0,
              "line %d is [%s] and [%s] was asked for", i, lines[i], want);
        snprintf(tail, sizeof(tail), " sampled=%d replayed=%d",
                 ((token.flags & AOTX_TOKEN_SAMPLED) != 0) ? 1 : 0, replayed);
        CHECK(strstr(lines[i], tail) != NULL, "line %d does not state both flags", i);
    }
}

static void batch(int n)
{
    char dir[256];
    char boot_dir[320];
    char out_path[1024];
    uint64_t boot_id = 0x00000000face0001ull;
    int count;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    build_journal(dir, boot_id, n, 0);
    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    snprintf(out_path, sizeof(out_path), "%s/tokens.txt", dir);

    /* A boot directory is read as it stands. */
    CHECK(run_reader(boot_dir, NULL, out_path) == 0, "the reader does not end with a clean status");
    count = split(out_path);
    compare(count, n, 0);

    /* A journal directory with no boot named gives the newest boot. */
    CHECK(run_reader(dir, NULL, out_path) == 0, "the reader refuses a journal directory");
    count = split(out_path);
    compare(count, n, 0);

    /* A journal directory with a boot named gives that boot. */
    {
        char name[32];
        snprintf(name, sizeof(name), "%016llx", (unsigned long long)boot_id);
        CHECK(run_reader(dir, name, out_path) == 0, "the reader refuses a named boot");
        count = split(out_path);
        compare(count, n, 0);
    }
    printf("batch %d: token lines %d\n", n, count);
    aotx_remove_tree(dir);
}

/* Checks the choice between two boots, the replayed field, and the refusals. */
static void choices(void)
{
    char dir[256];
    char out_path[1024];
    char name[32];
    int count;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(out_path, sizeof(out_path), "%s/tokens.txt", dir);

    /* A directory with no boot in it holds no journal. */
    CHECK(run_reader(dir, NULL, out_path) == AOTX_EXIT_NOJOURNAL,
          "an empty directory must give the no journal status");
    CHECK(run_reader(dir, "0000000000000009", out_path) == AOTX_EXIT_NOJOURNAL,
          "a boot that is not there must give the no journal status");

    build_journal(dir, 0x00000000face0001ull, 3, 0);
    build_journal(dir, 0x00000000face0002ull, 5, 1);

    /* The boot record wall clock of the second journal is the later one. */
    CHECK(run_reader(dir, NULL, out_path) == 0, "the reader does not end with a clean status");
    count = split(out_path);
    compare(count, 5, 1);

    snprintf(name, sizeof(name), "%016llx", 0x00000000face0001ull);
    CHECK(run_reader(dir, name, out_path) == 0, "the reader refuses the first boot");
    count = split(out_path);
    compare(count, 3, 0);
    printf("choices: newest boot 5 lines, named boot %d lines\n", count);
    aotx_remove_tree(dir);
}

/* A command that the reader does not know must give a fault status and no line. */
static void refusals(void)
{
    char dir[256];
    char out_path[1024];
    char *args[4];
    int child;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(out_path, sizeof(out_path), "%s/tokens.txt", dir);
    args[0] = arguments[1];
    args[1] = (char *)"records";
    args[2] = dir;
    args[3] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the reader does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_FAULT, "an unknown command must give the fault status");
    args[1] = (char *)"tokens";
    args[2] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the reader does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_FAULT, "a command with no directory must be refused");
    aotx_remove_tree(dir);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 2) {
        printf("usage: journal_test <journal reader program>\n");
        return 1;
    }
    batch(1);
    batch(64);
    choices();
    refusals();
    return aotx_report("journal_test", 40);
}
