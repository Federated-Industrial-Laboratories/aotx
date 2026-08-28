/* Purpose: Check the segment frames round trip and that damage is refused.
 * Owns: One temporary directory for each case.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>
#include <unistd.h>

#define AOTX_BUFFER_BYTES 8192

/* Builds one block whose records carry content that no other block carries. */
static uint32_t build(aotx_fake_device *d, unsigned char *out, int index, int records)
{
    char body[64];
    int i;
    for (i = 0; i < records; i++) {
        snprintf(body, sizeof(body), "block %d record %d", index, i);
        aotx_fake_record(d, AOTX_CLASS_B, AOTX_REC_CONSOLE, body, (uint32_t)strlen(body));
    }
    aotx_fake_commit(d, 0);
    memcpy(out, d->stage, ((const aotx_block_header *)d->stage)->byte_len);
    return ((const aotx_block_header *)d->stage)->byte_len;
}

/* Writes n blocks, reads them back, and compares every byte. */
static void round_trip(int n, uint64_t limit, int expect_segments)
{
    aotx_fake_device device;
    aotx_segment_writer writer;
    char dir[256];
    static unsigned char written[64][AOTX_BUFFER_BYTES];
    static unsigned char got[AOTX_BUFFER_BYTES];
    uint32_t lengths[64];
    int segments;
    int index = 0;
    int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    aotx_fake_start(&device, NULL, 0x1122334455667788ull);
    CHECK(aotx_segment_open(&writer, dir, limit) == 0, "the segment does not open");
    for (i = 0; i < n; i++) {
        lengths[i] = build(&device, written[i], i, 1 + (i % 7));
        CHECK(aotx_segment_put(&writer, written[i], lengths[i]) == 0, "the frame does not write");
    }
    CHECK(aotx_segment_close(&writer) == 0, "the segment does not close");

    segments = aotx_segment_list(dir, NULL, 0);
    CHECK(segments >= expect_segments, "found %d segments and %d were asked for",
          segments, expect_segments);
    {
        char names[64][AOTX_NAME_BYTES];
        int count = aotx_segment_list(dir, names, 64);
        for (i = 0; i < count; i++) {
            char path[1024];
            aotx_segment_reader reader;
            snprintf(path, sizeof(path), "%s/%.*s", dir, AOTX_NAME_BYTES - 1, names[i]);
            CHECK(aotx_segment_reader_open(&reader, path) == 0, "the segment does not read");
            for (;;) {
                uint32_t bytes = 0;
                int frame = aotx_segment_get(&reader, got, sizeof(got), &bytes);
                if (frame == AOTX_FRAME_END) {
                    break;
                }
                CHECK(frame == AOTX_FRAME_OK, "frame %d is not whole", index);
                CHECK(index < n, "the file holds more frames than were written");
                if (index < n) {
                    CHECK(bytes == lengths[index], "frame %d has the wrong length", index);
                    CHECK(memcmp(got, written[index], bytes) == 0, "frame %d differs", index);
                }
                index++;
            }
            aotx_segment_reader_close(&reader);
        }
    }
    CHECK(index == n, "read %d frames and wrote %d", index, n);
    aotx_remove_tree(dir);
}

/* Writes three frames, damages the second, and checks the reader stops at the damage. */
static void damage(int flip, int cut)
{
    aotx_fake_device device;
    aotx_segment_writer writer;
    aotx_segment_reader reader;
    static unsigned char block[AOTX_BUFFER_BYTES];
    static unsigned char got[AOTX_BUFFER_BYTES];
    char dir[256];
    char path[1024];
    uint32_t first_len;
    int fd;
    int i;
    uint32_t bytes = 0;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    aotx_fake_start(&device, NULL, 7);
    CHECK(aotx_segment_open(&writer, dir, 0) == 0, "the segment does not open");
    first_len = build(&device, block, 0, 3);
    CHECK(aotx_segment_put(&writer, block, first_len) == 0, "the first frame does not write");
    for (i = 1; i < 3; i++) {
        uint32_t len = build(&device, block, i, 3);
        CHECK(aotx_segment_put(&writer, block, len) == 0, "a later frame does not write");
    }
    CHECK(aotx_segment_close(&writer) == 0, "the segment does not close");

    snprintf(path, sizeof(path), "%s/seg-000000.seg", dir);
    if (flip) {
        unsigned char byte = 0;
        off_t at = (off_t)(8 + first_len + 8 + 80);
        fd = open(path, O_RDWR);
        CHECK(fd >= 0, "the segment does not open for the change");
        CHECK(pread(fd, &byte, 1, at) == 1, "the byte does not read");
        byte = (unsigned char)(byte ^ 0x40u);
        CHECK(pwrite(fd, &byte, 1, at) == 1, "the byte does not write");
        close(fd);
    }
    if (cut) {
        struct stat st;
        CHECK(stat(path, &st) == 0, "the segment does not measure");
        CHECK(truncate(path, st.st_size - 17) == 0, "the segment does not shorten");
    }
    CHECK(aotx_segment_reader_open(&reader, path) == 0, "the segment does not read");
    CHECK(aotx_segment_get(&reader, got, sizeof(got), &bytes) == AOTX_FRAME_OK,
          "the frame before the damage must still read");
    CHECK(bytes == first_len, "the frame before the damage has the wrong length");
    if (flip) {
        CHECK(aotx_segment_get(&reader, got, sizeof(got), &bytes) == AOTX_FRAME_TORN,
              "a changed byte must fail the checksum");
    } else {
        int frame = AOTX_FRAME_OK;
        while (frame == AOTX_FRAME_OK) {
            frame = aotx_segment_get(&reader, got, sizeof(got), &bytes);
        }
        CHECK(frame == AOTX_FRAME_TORN, "a short file must end as torn");
    }
    aotx_segment_reader_close(&reader);
    aotx_remove_tree(dir);
}

int main(void)
{
    round_trip(1, 0, 1);
    round_trip(64, 0, 1);
    /* A limit near one frame makes the writer open a new segment for almost every block. */
    round_trip(64, 2048, 8);
    damage(1, 0);
    damage(0, 1);
    return aotx_report("segment_test", 200);
}
