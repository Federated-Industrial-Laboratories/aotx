/* Purpose: Run the stopped-process and console checks after device release.
 * Owns: The process identity and its exit status.
 * Threading: One host thread starts one process for each model file.
 * Lifetime: One architecture check. */
#ifndef AOTX_TEST_ARCH_PROCESS_H
#define AOTX_TEST_ARCH_PROCESS_H

static void aotx_arch_load_line(const aotx_model_desc *desc, char *out, size_t room)
{
    size_t used = (size_t)snprintf(out, room, "layers:");
    for (unsigned int first = 0u; first < desc->layers;) {
        unsigned int last = first + 1u;
        while (last < desc->layers && desc->kind[last] == desc->kind[first]) ++last;
        const aotx_layer_kind *kind = aotx_layer_kind_of(desc->kind[first]);
        if (used >= room || kind == NULL) { out[0] = '\0'; return; }
        int n = snprintf(out + used, room - used, " %u %s", last - first, kind->name);
        if (n < 0 || (size_t)n >= room - used) { out[0] = '\0'; return; }
        used += (size_t)n;
        first = last;
    }
}

/* The script returns a bit for each failed check: restore 1, load line 2, setup 4. */
static void aotx_arch_process(const char *program, const char *script, const char *out,
                               const char *models, const aotx_manifest_entry *entry,
                               const char *load_line)
{
    char build[PATH_MAX];
    int status = 4;
    if (realpath(program, build) != NULL) {
        char *slash = strrchr(build, '/');
        if (slash != NULL) *slash = '\0';
        fflush(NULL);
        pid_t child = fork();
        if (child == 0) {
            setpgid(0, 0);
            execlp("python3", "python3", script, "--build", build, "--store", models,
                   "--name", entry->name, "--load-line", load_line, "--out", out,
                   (char *)NULL);
            _exit(4);
        }
        if (child > 0) {
            setpgid(child, child);
            time_t until = time(NULL) + 3600;
            int result = 0;
            for (;;) {
                pid_t got = waitpid(child, &result, WNOHANG);
                if (got == child) {
                    status = WIFEXITED(result) ? WEXITSTATUS(result) : 4;
                    break;
                }
                if (got < 0 && errno != EINTR) break;
                if (time(NULL) >= until) {
                    kill(-child, SIGKILL);
                    waitpid(child, &result, 0);
                    printf("arch: %s process check timed out\n", entry->path);
                    break;
                }
                usleep(100000);
            }
        }
    }
    printf("arch: %s process exit %d\n", entry->path, status);
    int valid = status >= 0 && status <= 3;
    aotx_arch_check(valid && !(status & 1), entry->path,
                    "5 restore: stopped third turn matches continuous third turn");
    aotx_arch_check(valid && !(status & 2), entry->path,
                    "6 load line: console layer sequence matches descriptor");
}
#endif
