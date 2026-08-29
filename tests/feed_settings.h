/* Purpose: Give the feeder check the arm of the settings that go out before every record.
 * Owns: Nothing; the check that includes this file holds the ring and the child process.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: One case of the check.
 *
 * The file is included by tests/feed_test.c after the parts that every arm shares. */
#ifndef AOTX_TESTS_FEED_SETTINGS_H
#define AOTX_TESTS_FEED_SETTINGS_H

#define AOTX_SETTINGS_TEXT 8192

typedef struct settings_taken {
    int      count;             /* setting records that came */
    int      lines;             /* input line records that came */
    int      clocks;
    uint64_t first_line_seq;    /* the sequence of the first input line record */
    uint64_t first_clock_seq;   /* the sequence of the first clock record */
    uint64_t last_setting_seq;
    aotx_setting_body body[AOTX_SETTING_NUMBER_COUNT];
} settings_taken;

/* Consumes the slots that the feeder published and keeps the setting records in order. */
static void take_settings(aotx_inbound_ring *ring, settings_taken *t)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
        if (h->type == AOTX_REC_SETTING) {
            CHECK(h->cls == AOTX_CLASS_A, "a setting record is not authoritative");
            CHECK(h->writer == AOTX_WRITER_FEEDER, "a setting record holds the wrong writer");
            CHECK(h->body_len == sizeof(aotx_setting_body),
                  "a setting record has the wrong body length");
            if (t->count < (int)AOTX_SETTING_NUMBER_COUNT) {
                memcpy(&t->body[t->count], aotx_record_body(h), sizeof(aotx_setting_body));
            }
            t->last_setting_seq = h->seq;
            t->count++;
        } else if (h->type == AOTX_REC_INPUT_LINE) {
            if (t->lines == 0) {
                t->first_line_seq = h->seq;
            }
            t->lines++;
        } else if (h->type == AOTX_REC_TICK_START) {
            if (t->clocks == 0) {
                t->first_clock_seq = h->seq;
            }
            t->clocks++;
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Gives the value that line i of the settings file writes for the device key at place k of
 * the device key list. A key that comes again takes another value, so a file where the
 * last line does not win cannot pass. */
static int64_t settings_value(unsigned int key, int occurrence)
{
    int64_t least = aotx_settings_number_least(key);
    int64_t span = aotx_settings_number_most(key) - least + 1;
    return least + ((int64_t)occurrence * 3 + (int64_t)key) % span;
}

/* Runs the feeder with a settings file of n lines. The records must reach the ring before
 * the first line of the standard input. They must also come before the first clock record.
 * They come in the order of the key list. Each holds the value of the last line of its
 * key. */
static void settings_arm(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    settings_taken got;
    unsigned int device[AOTX_SETTING_NUMBER_COUNT];
    int last[AOTX_SETTING_NUMBER_COUNT];
    char dir[256];
    char path[320];
    char file_text[AOTX_SETTINGS_TEXT];
    char err_path[400];
    char value[32];
    char fd_text[16];
    char *args[6];
    unsigned int k;
    size_t used;
    int device_count = 0;
    int want;
    int pipe_fds[2];
    int err_fd;
    int saved;
    int child;
    int i;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    for (k = 0; k < AOTX_SETTING_NUMBER_COUNT; k++) {
        last[k] = -1;
        if (aotx_settings_number_side(k) == AOTX_SETTING_SIDE_DEVICE) {
            device[device_count++] = k;
        }
    }
    CHECK(device_count > 0, "the key list names no device side number key");
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);

    /* Line 1 is a comment and line 2 is blank, so the refused line is line 3. */
    used = (size_t)snprintf(file_text, sizeof(file_text),
                            "# the settings of a run of %d lines\n\nno.such.key = 1\n", n);
    for (i = 0; i < n; i++) {
        k = device[i % device_count];
        last[k] = i / device_count;
        aotx_settings_format(settings_value(k, i / device_count),
                             aotx_settings_number_scale(k), value, sizeof(value));
        used += (size_t)snprintf(file_text + used, sizeof(file_text) - used, "  %s  =  %s  \n",
                                 aotx_settings_number_name(k), value);
    }
    /* A boot key and a text key make no record, because the device applies neither. */
    used += (size_t)snprintf(file_text + used, sizeof(file_text) - used,
                             "window.on = 1\ntui.box = unicode\n");
    {
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        CHECK(fd >= 0, "the settings file does not open");
        CHECK(write(fd, file_text, used) == (ssize_t)used, "the settings file does not write");
        close(fd);
    }
    want = (n < device_count) ? n : device_count;

    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    /* The line waits in the pipe before the feeder starts. A feeder that reads the
     * standard input first cannot pass this case. */
    CHECK(write(pipe_fds[1], "the first operator line\n", 24) == 24,
          "the operator line does not write");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--settings";
    args[4] = path;
    args[5] = NULL;
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
    close(pipe_fds[1]);

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while ((got.count < want || got.lines < 1 || got.clocks < 1) &&
           aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        take_settings(&ring, &got);
        aotx_pause(&backoff);
    }
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    take_settings(&ring, &got);

    CHECK(got.count == want, "the feeder sent %d setting records and %d were asked for",
          got.count, want);
    CHECK(got.lines >= 1 && got.clocks >= 1, "the run gave %d lines and %d clock records",
          got.lines, got.clocks);
    CHECK(got.last_setting_seq < got.first_line_seq, "a setting record came after the first"
          " line of the standard input");
    CHECK(got.last_setting_seq < got.first_clock_seq, "a setting record came after the first"
          " clock record");
    for (i = 0; i < want && i < got.count; i++) {
        const aotx_setting_body *b = &got.body[i];
        char name[AOTX_SETTING_WIRE_KEY_BYTES + 1];
        k = device[i];
        memcpy(name, b->key, b->key_len);
        name[b->key_len] = '\0';
        CHECK(strcmp(name, aotx_settings_number_name(k)) == 0,
              "setting record %d names %s and %s was asked for", i, name,
              aotx_settings_number_name(k));
        CHECK(b->scale == (uint32_t)aotx_settings_number_scale(k),
              "setting record %d holds the scale %u", i, b->scale);
        CHECK(b->value == settings_value(k, last[k]), "setting record %d holds %lld and the"
              " last line of %s gives %lld", i, (long long)b->value,
              aotx_settings_number_name(k), (long long)settings_value(k, last[k]));
    }
    /* The start prints a refused line; the feeder prints nothing, so a refusal reaches the
     * operator one time. */
    {
        char report[4096];
        int fd = open(err_path, O_RDONLY);
        ssize_t bytes = 0;
        CHECK(fd >= 0, "the report file does not read");
        if (fd >= 0) {
            bytes = read(fd, report, sizeof(report) - 1);
            close(fd);
        }
        report[(bytes > 0) ? bytes : 0] = '\0';
        CHECK(strstr(report, "settings:") == NULL,
              "the feeder must print no refusal; the start prints it: %s", report);
    }
    printf("settings %d: records %d, lines in the file %d\n", n, got.count, n + 5);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

#endif
