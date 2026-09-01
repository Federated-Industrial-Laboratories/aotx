// Purpose: End one host child with a bounded escalation sequence.
// Owns: No child; the caller owns the process identity and result text.
// Launch shape: One interface thread waits for at most one bounded interval.
// Lifetime: The helper returns after the child is reaped.
#ifndef AOTX_CTRL_PROCESS_CHILD_HPP
#define AOTX_CTRL_PROCESS_CHILD_HPP

#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <thread>

namespace aotx::ctrl::process {

enum class End { absent, term, kill };

inline End end_child(pid_t child)
{
    if (child < 0) return End::absent;
    if (::kill(child, SIGTERM) != 0 && errno == ESRCH) {
        while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
        return End::term;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t ended = ::waitpid(child, nullptr, WNOHANG);
        if (ended == child || (ended < 0 && errno == ECHILD)) return End::term;
        if (ended < 0 && errno != EINTR) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    (void)::kill(child, SIGKILL);
    while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    return End::kill;
}

} // namespace aotx::ctrl::process

#endif
