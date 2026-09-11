/* Purpose: Run the complete file test with IP socket creation disabled by the kernel.
 * Owns: A process filter inherited by the program and all children.
 * Threading: One launcher installs the filter before exec.
 * Lifetime: The filter stays active until the last child exits. */
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) { fputs("Use: aotx_runtime_offline PROGRAM ARG...\n", stderr); return 2; }
#if defined(__x86_64__)
    const unsigned architecture = AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
    const unsigned architecture = AUDIT_ARCH_AARCH64;
#else
#error "the offline test requires a supported Linux architecture"
#endif
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, architecture, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_INET, 1, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_INET6, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
    };
    struct sock_fprog program = {(unsigned short)(sizeof(filter) / sizeof(filter[0])), filter};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) || prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program)) {
        perror("offline filter"); return 1;
    }
    int domains[] = {AF_INET, AF_INET6};
    for (unsigned i = 0; i < sizeof(domains) / sizeof(domains[0]); ++i) {
        if (socket(domains[i], SOCK_STREAM, 0) != -1 || errno != EPERM) {
            fputs("the kernel did not refuse an IP socket\n", stderr); return 1;
        }
    }
    fputs("network: IPv4 and IPv6 sockets are disabled\n", stderr);
    execvp(argv[1], argv + 1); perror("offline exec"); return 1;
}
