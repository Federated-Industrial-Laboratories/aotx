/* Purpose: Verify complete replacement, lease transfer and bounded crash recovery.
 * Owns: Distinct optional section batches and syscall fault controls.
 * Threading: One writer child at each publication boundary.
 * Lifetime: Each case removes its temporary files. */
#include "tests/ccir_disk_fixture.h"
#include <dirent.h>
#include <sys/wait.h>

static int active, operation, target, mode;
ssize_t __real_pwrite(int, const void *, size_t, off_t);
int __real_fsync(int);
int __real_rename(const char *, const char *);
int __real_linkat(int, const char *, int, const char *, int);
static int before(void) {
    if (!active) return 0;
    ++operation;
    if (operation != target) return 0;
    if (mode == 1) { errno = ENOSPC; return 1; }
    if (mode == 2) _exit(91);
    return 0;
}
static void after(void) { if (active && operation == target && mode == 3) _exit(91); }
ssize_t __wrap_pwrite(int fd, const void *p, size_t n, off_t at) {
    if (before()) return -1;
    ssize_t result = __real_pwrite(fd, p, n, at); after(); return result;
}
int __wrap_fsync(int fd) {
    if (before()) return -1;
    int result = __real_fsync(fd); after(); return result;
}
int __wrap_rename(const char *a, const char *b) {
    if (before()) return -1;
    int result = __real_rename(a, b); after(); return result;
}
int __wrap_linkat(int a, const char *b, int c, const char *d, int flags) {
    if (before()) return -1;
    int result = __real_linkat(a, b, c, d, flags); after(); return result;
}
static unsigned pending_count(const char *directory) {
    DIR *dir = opendir(directory); struct dirent *entry; unsigned count = 0;
    CHECK(dir != NULL); if (!dir) return 99;
    while ((entry = readdir(dir))) if (strstr(entry->d_name, ".pending-")) ++count;
    closedir(dir); return count;
}
static void verify(aotx_ccir_view *v, const aotx_ccir_fixture *f, uint64_t first, uint64_t second) {
    CHECK(v->generation == 1 && v->count == f->count);
    CHECK(v->meta.durable_sequence == first || v->meta.durable_sequence == second);
    for (uint32_t i = 0; i < f->count; ++i) {
        unsigned char p[512]; const aotx_ccir_section *s = &v->sections[i];
        CHECK(!memcmp(s->id, f->inputs[i].section.id, 16) && s->type == f->inputs[i].section.type &&
              s->schema == f->inputs[i].section.schema && s->bytes == f->inputs[i].section.bytes);
        CHECK(aotx_ccir_pread(v->fd, p, (size_t)s->bytes, s->offset) == 0);
        CHECK(!memcmp(p, f->inputs[i].data, (size_t)s->bytes));
    }
}
static void run(const char *directory, const char *base, const char *path, uint32_t n) {
    aotx_ccir_fixture f; fixture(&f, n);
    CHECK(aotx_ccir_create(base, f.lineage, f.inputs, f.count, &f.meta, NULL) == 0);
    f.meta.checkpoint_sequence++; f.meta.durable_sequence++;
    CHECK(copy_file(base, path) == 0);
    aotx_ccir_view view, other;
    CHECK(aotx_ccir_writer_open(path, NULL, &view) == 0);
    CHECK(aotx_ccir_open(path, NULL, &other) == 0);
    CHECK(aotx_ccir_writer_replace(&view, path, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_BUSY);
    aotx_ccir_close(&other);
    aotx_ccir_input batch[66]; memcpy(batch, f.inputs, sizeof(batch));
    for (uint32_t i = 2; i < f.count; ++i) { batch[i].source = AOTX_CCIR_REUSE; batch[i].data = NULL; }
    active = 1; operation = target = mode = 0;
    int status = aotx_ccir_writer_replace(&view, path, batch, f.count, &f.meta, NULL);
    active = 0; int total = operation;
    CHECK(status == 0); verify(&view, &f, n + 1, n + 1);
    CHECK(aotx_ccir_writer_open(path, NULL, &other) == AOTX_CCIR_BUSY);
    CHECK(aotx_ccir_open(path, NULL, &other) == 0); aotx_ccir_close(&other);
    aotx_ccir_close(&view); CHECK(pending_count(directory) == 0); unlink(path);
    for (int action = 1; action <= 3; ++action) for (int point = 1; point <= total; ++point) {
        CHECK(copy_file(base, path) == 0);
        if (action == 1) {
            CHECK(aotx_ccir_writer_open(path, NULL, &view) == 0);
            active = 1; operation = 0; target = point; mode = action;
            status = aotx_ccir_writer_replace(&view, path, batch, f.count, &f.meta, NULL);
            active = 0; CHECK(status == AOTX_CCIR_IO);
            verify(&view, &f, n, n + 1);
            CHECK(aotx_ccir_writer_open(path, NULL, &other) == AOTX_CCIR_BUSY);
            CHECK(aotx_ccir_writer_sync(&view, path) == 0);
            aotx_ccir_close(&view);
        } else {
            pid_t child = fork(); CHECK(child >= 0);
            if (!child) {
                if (aotx_ccir_writer_open(path, NULL, &view)) _exit(92);
                active = 1; operation = 0; target = point; mode = action;
                (void)aotx_ccir_writer_replace(&view, path, batch, f.count, &f.meta, NULL);
                _exit(93);
            }
            int exit_status = 0;
            CHECK(waitpid(child, &exit_status, 0) == child);
            CHECK(WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 91);
            CHECK(pending_count(directory) <= 1);
        }
        CHECK(aotx_ccir_writer_open(path, NULL, &view) == 0);
        verify(&view, &f, n, n + 1); CHECK(pending_count(directory) == 0);
        CHECK(aotx_ccir_writer_replace(&view, path, batch, f.count, &f.meta, NULL) == 0);
        verify(&view, &f, n + 1, n + 1); aotx_ccir_close(&view);
        CHECK(pending_count(directory) == 0); unlink(path);
    }
    CHECK(copy_file(base, path) == 0);
    CHECK(aotx_ccir_writer_open(path, NULL, &view) == 0);
    char pending[512]; size_t at = strlen(path);
    memcpy(pending, path, at); memcpy(pending + at, ".pending-", 9);
    for (unsigned i = 0; i < 16; ++i) snprintf(pending + at + 9 + i * 2, 3, "%02x", view.incarnation[i]);
    int fd = open(pending, O_WRONLY | O_CREAT | O_EXCL, 0600); CHECK(fd >= 0);
    CHECK(write(fd, "unrelated", 9) == 9); close(fd);
    CHECK(aotx_ccir_writer_replace(&view, path, batch, f.count, &f.meta, NULL) != 0);
    struct stat st; CHECK(!stat(pending, &st) && st.st_size == 9);
    verify(&view, &f, n, n); aotx_ccir_close(&view); unlink(pending); unlink(path); unlink(base);
    printf("maintenance disk N=%u: %d syscall boundaries\n", n, total);
}
int main(void) {
    char directory[] = "/tmp/aotx-maintenance-disk-XXXXXX", base[256], path[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(base, sizeof(base), "%s/base", directory); snprintf(path, sizeof(path), "%s/state", directory);
    run(directory, base, path, 1); run(directory, base, path, 64);
    CHECK(rmdir(directory) == 0);
    printf("maintenance disk: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
