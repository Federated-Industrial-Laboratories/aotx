/* Purpose: Check short writes, disk faults and process loss at publication boundaries.
 * Owns: Syscall wrappers and temporary source and destination files.
 * Threading: Each child writes alone; the parent checks the released file.
 * Lifetime: A case removes all temporary files after verification. */
#include "tests/ccir_disk_fixture.h"
#include <sys/wait.h>

static int aotx_active, aotx_mode, aotx_target;
static int aotx_operations, aotx_writes, aotx_syncs, aotx_parent_syncs;
static int aotx_order_errors;
ssize_t __real_pwrite(int fd, const void *data, size_t bytes, off_t offset);
int __real_fsync(int fd);

ssize_t __wrap_pwrite(int fd, const void *data, size_t bytes, off_t offset)
{
    ssize_t n;
    if (!aotx_active) return __real_pwrite(fd, data, bytes, offset);
    aotx_operations++; aotx_writes++;
    if (aotx_mode == 0 && bytes >= 8u) {
        if (!memcmp(data, "AOTXCMT1", 8u) && aotx_syncs != 1) aotx_order_errors++;
        if (!memcmp(data, "AOTXROOT", 8u) && aotx_syncs != 2) aotx_order_errors++;
    }
    if (aotx_mode == 4 && aotx_operations == aotx_target) _exit(91);
    if (aotx_mode == 2 && aotx_writes == aotx_target) { errno = ENOSPC; return -1; }
    if (aotx_mode == 6 && offset == 8192) {
        n = __real_pwrite(fd, data, bytes / 2u, offset);
        if (n != (ssize_t)(bytes / 2u)) _exit(93);
        errno = ENOSPC; return -1;
    }
    if (aotx_mode == 1 && bytes > 7u) bytes = 7u;
    n = __real_pwrite(fd, data, bytes, offset);
    if (aotx_mode == 5 && aotx_operations == aotx_target) _exit(91);
    return n;
}
int __wrap_fsync(int fd)
{
    int rc;
    struct stat st;
    if (!aotx_active) return __real_fsync(fd);
    aotx_operations++; aotx_syncs++;
    if (!fstat(fd, &st) && S_ISDIR(st.st_mode)) {
        aotx_parent_syncs++;
        if (aotx_mode == 0 && aotx_syncs != 4) aotx_order_errors++;
    }
    if (aotx_mode == 4 && aotx_operations == aotx_target) _exit(91);
    if (aotx_mode == 3 && aotx_syncs == aotx_target) { errno = EIO; return -1; }
    rc = __real_fsync(fd);
    if (aotx_mode == 5 && aotx_operations == aotx_target) _exit(91);
    return rc;
}
static void arm(int mode, int target)
{
    aotx_mode = mode; aotx_target = target;
    aotx_operations = aotx_writes = aotx_syncs = aotx_parent_syncs = aotx_order_errors = 0;
    aotx_active = 1;
}
static void complete(const char *path, uint64_t first, uint64_t second)
{
    aotx_ccir_view view;
    int rc = aotx_ccir_open(path, NULL, &view);
    CHECK(rc == AOTX_CCIR_OK);
    if (!rc) {
        CHECK(view.generation == first || view.generation == second);
        CHECK(view.meta.durable_sequence == 64u + view.generation - 1u ||
              view.meta.durable_sequence == 1u + view.generation - 1u);
        aotx_ccir_close(&view);
    }
}
static void interrupted(const char *path, const char *base, const char *destination,
                        aotx_ccir_fixture *f, int operation, int mode, int target)
{
    pid_t child;
    int status;
    CHECK(copy_file(base, path) == 0);
    unlink(destination);
    child = fork(); CHECK(child >= 0);
    if (child == 0) {
        arm(mode, target);
        if (operation == 0) (void)aotx_ccir_append(path, f->inputs, f->count, &f->meta, NULL);
        if (operation == 1) (void)aotx_ccir_create(destination, f->lineage, f->inputs,
                                                 f->count, &f->meta, NULL);
        if (operation == 2) (void)aotx_ccir_compact(path, destination, NULL);
        _exit(92);
    }
    if (child < 0) return;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 91);
    if (operation == 0) complete(path, 1u, 2u);
    else {
        aotx_ccir_view view;
        int rc;
        CHECK(reopen_generation(path, 1u));
        rc = aotx_ccir_open(destination, NULL, &view);
        CHECK(rc == AOTX_CCIR_OK || rc == AOTX_CCIR_INVALID || rc == AOTX_CCIR_IO);
        if (!rc) { CHECK(view.generation == 1u); aotx_ccir_close(&view); }
    }
    unlink(path); unlink(destination);
}
static void faults(const char *base, const char *path, const char *destination, uint32_t n)
{
    aotx_ccir_fixture f;
    int rc, writes, syncs, operations, create_operations, compact_operations, i, mode;
    unsigned char original_digest[32], final_digest[32];
    struct stat st;
    int source_fd;
    fixture(&f, n);
    CHECK(aotx_ccir_create(base, f.lineage, f.inputs, f.count, &f.meta, NULL) == 0);
    source_fd = open(base, O_RDONLY); CHECK(source_fd >= 0);
    CHECK(fstat(source_fd, &st) == 0);
    CHECK(aotx_ccir_hash_fd(source_fd, 0u, (uint64_t)st.st_size, original_digest) == 0);
    f.meta.checkpoint_sequence++; f.meta.durable_sequence++; f.payload[0][0] ^= 71u;
    CHECK(copy_file(base, path) == 0);
    arm(0, 0);
    rc = aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL);
    aotx_active = 0;
    writes = aotx_writes; syncs = aotx_syncs; operations = aotx_operations;
    CHECK(rc == 0 && writes == (int)f.count + 3 && syncs == 3);
    CHECK(aotx_order_errors == 0);
    CHECK(aotx_parent_syncs == 0);
    unlink(path);
    CHECK(copy_file(base, path) == 0);
    arm(1, 0); rc = aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL);
    aotx_active = 0;
    CHECK(rc == 0); complete(path, 2u, 2u); unlink(path);
    for (i = 1; i <= writes; i++) {
        CHECK(copy_file(base, path) == 0);
        arm(2, i); rc = aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL);
        aotx_active = 0; CHECK(rc == AOTX_CCIR_IO);
        complete(path, 1u, 1u); unlink(path);
    }
    for (i = 1; i <= syncs; i++) {
        CHECK(copy_file(base, path) == 0);
        arm(3, i); rc = aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL);
        aotx_active = 0; CHECK(rc == AOTX_CCIR_IO);
        complete(path, 1u, 2u); unlink(path);
    }
    CHECK(copy_file(base, path) == 0);
    arm(6, 0); rc = aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL);
    aotx_active = 0; CHECK(rc == AOTX_CCIR_IO);
    complete(path, 1u, 1u); unlink(path);
    arm(0, 0);
    rc = aotx_ccir_create(destination, f.lineage, f.inputs, f.count, &f.meta, NULL);
    aotx_active = 0; create_operations = aotx_operations;
    CHECK(rc == 0 && aotx_parent_syncs == 1 && aotx_syncs == 4);
    CHECK(aotx_order_errors == 0);
    unlink(destination);
    for (mode = 2; mode <= 3; mode++) {
        int total = mode == 2 ? create_operations - 4 : 4;
        for (i = 1; i <= total; i++) {
            arm(mode, i);
            rc = aotx_ccir_create(destination, f.lineage, f.inputs, f.count, &f.meta, NULL);
            aotx_active = 0;
            CHECK(rc == AOTX_CCIR_IO && access(destination, F_OK) != 0);
        }
    }
    arm(3, 4);
    rc = aotx_ccir_create(destination, f.lineage, f.inputs, f.count, &f.meta, NULL);
    aotx_active = 0; CHECK(rc == AOTX_CCIR_IO && access(destination, F_OK) != 0);
    arm(0, 0); rc = aotx_ccir_compact(base, destination, NULL);
    aotx_active = 0; compact_operations = aotx_operations;
    CHECK(rc == 0 && aotx_parent_syncs == 1 && aotx_syncs == 4);
    CHECK(aotx_order_errors == 0);
    unlink(destination);
    arm(1, 0); rc = aotx_ccir_compact(base, destination, NULL);
    aotx_active = 0; CHECK(rc == AOTX_CCIR_OK && reopen_generation(destination, 1u));
    unlink(destination);
    for (mode = 2; mode <= 3; mode++) {
        int total = mode == 2 ? compact_operations - 4 : 4;
        for (i = 1; i <= total; i++) {
            arm(mode, i); rc = aotx_ccir_compact(base, destination, NULL);
            aotx_active = 0;
            CHECK(rc == AOTX_CCIR_IO && access(destination, F_OK) != 0);
            CHECK(reopen_generation(base, 1u));
        }
    }
    for (mode = 4; mode <= 5; mode++) {
        for (i = 1; i <= operations; i++) interrupted(path, base, destination, &f, 0, mode, i);
        for (i = 1; i <= create_operations; i++) interrupted(path, base, destination, &f, 1, mode, i);
        for (i = 1; i <= compact_operations; i++) interrupted(path, base, destination, &f, 2, mode, i);
    }
    printf("ccir faults N=%u: append %d, create %d, compact %d IO boundaries\n",
           n, operations, create_operations, compact_operations);
    CHECK(aotx_ccir_hash_fd(source_fd, 0u, (uint64_t)st.st_size, final_digest) == 0);
    CHECK(!memcmp(original_digest, final_digest, 32u));
    {
        struct stat after;
        CHECK(fstat(source_fd, &after) == 0 && after.st_size == st.st_size);
    }
    close(source_fd);
    unlink(base);
}
int main(void)
{
    char directory[] = "/tmp/aotx-ccir-fault-XXXXXX", base[256], path[256], destination[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(base, sizeof(base), "%s/base.aotxccir", directory);
    snprintf(path, sizeof(path), "%s/state.aotxccir", directory);
    snprintf(destination, sizeof(destination), "%s/copy.aotxccir", directory);
    faults(base, path, destination, 1u); faults(base, path, destination, 64u);
    rmdir(directory);
    printf("ccir disk faults: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
