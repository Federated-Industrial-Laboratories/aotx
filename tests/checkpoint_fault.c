/* Purpose: Inject bounded write and sync failures into checkpoint file tests.
 * Owns: Test-only fault countdowns; normal calls reach the operating system.
 * Threading: One test writer.
 * Lifetime: Each explicit fault interval. */
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
static int writes = -1, syncs = -1;
void aotx_checkpoint_fault(int write_after, int sync_after) { writes = write_after; syncs = sync_after; }
ssize_t __real_pwrite(int fd, const void *data, size_t bytes, off_t offset);
int __real_fsync(int fd);
ssize_t __wrap_pwrite(int fd, const void *data, size_t bytes, off_t offset) {
    if (writes == 0) { errno = ENOSPC; return -1; }
    if (writes > 0) --writes;
    return __real_pwrite(fd, data, bytes, offset);
}
int __wrap_fsync(int fd) {
    if (syncs == 0) { errno = EIO; return -1; }
    if (syncs > 0) --syncs;
    return __real_fsync(fd);
}
