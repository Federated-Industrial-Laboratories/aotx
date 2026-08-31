// Purpose: Connect to the attach socket and exchange its exact frames.
// Owns: One socket, one received mirror descriptor, and frame buffers.
// Launch shape: One interface thread performs nonblocking socket work.
// Lifetime: Descriptors close with the client or after connection loss.
#include "client/client.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>

namespace aotx::ctrl::client {
namespace {

constexpr unsigned char attach_line = 'L';
constexpr unsigned char attach_reason = 'R';
constexpr unsigned char attach_mirror = 'M';
constexpr std::size_t line_bound = 6144u;
constexpr std::size_t incoming_bound = line_bound + 5u;
constexpr double retry_seconds = 2.0;

std::vector<unsigned char> line_frame(const std::string &line)
{
    const std::uint32_t length = static_cast<std::uint32_t>(line.size());
    std::vector<unsigned char> frame(5u + line.size());
    frame[0] = attach_line;
    frame[1] = static_cast<unsigned char>(length & 0xffu);
    frame[2] = static_cast<unsigned char>((length >> 8u) & 0xffu);
    frame[3] = static_cast<unsigned char>((length >> 16u) & 0xffu);
    frame[4] = static_cast<unsigned char>((length >> 24u) & 0xffu);
    std::copy(line.begin(), line.end(), frame.begin() + 5);
    return frame;
}

} // namespace

struct Client::Impl {
    explicit Impl(std::filesystem::path journal_path) : journal(std::move(journal_path)) {}

    enum class State { disconnected, attaching, connected };

    std::filesystem::path journal;
    int fd = -1;
    int mirror = -1;
    State state = State::disconnected;
    double retry_at = 0.0;
    std::vector<unsigned char> incoming;
    std::vector<unsigned char> outgoing;
    std::size_t sent = 0u;
    std::vector<std::string> results;
    bool outage_stated = false;

    ~Impl() { close_all(); }

    void close_all()
    {
        if (fd >= 0) ::close(fd);
        if (mirror >= 0) ::close(mirror);
        fd = -1;
        mirror = -1;
        state = State::disconnected;
        incoming.clear();
        outgoing.clear();
        sent = 0u;
    }

    /* One outage makes one line; the silent retries continue until the state changes. */
    void state_outage(const char *result)
    {
        if (outage_stated) return;
        outage_stated = true;
        results.emplace_back(result);
    }

    void lost(double now, const char *result)
    {
        close_all();
        retry_at = now + retry_seconds;
        state_outage(result);
    }

    void connect_now(double now)
    {
        sockaddr_un address{};
        const std::string path = (journal / "aotx.sock").string();
        if (path.size() >= sizeof(address.sun_path)) {
            retry_at = now + retry_seconds;
            state_outage("The socket path is too long.");
            return;
        }
        fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd < 0) {
            retry_at = now + retry_seconds;
            state_outage("The socket does not open.");
            return;
        }
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1u);
        if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 ||
            errno == EINPROGRESS) {
            state = State::attaching;
            return;
        }
        ::close(fd);
        fd = -1;
        retry_at = now + retry_seconds;
        state_outage("No running system was found at the journal. The connection starts with the system.");
    }

    void finish_connect(double now)
    {
        if (state != State::attaching) return;
        int error = 0;
        socklen_t bytes = sizeof(error);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &bytes) != 0 || error != 0) {
            lost(now, "The connection did not complete. A retry starts in 2 seconds.");
        }
    }

    void take_descriptor(msghdr &message)
    {
        for (cmsghdr *control = CMSG_FIRSTHDR(&message); control != nullptr;
             control = CMSG_NXTHDR(&message, control)) {
            if (control->cmsg_level != SOL_SOCKET || control->cmsg_type != SCM_RIGHTS ||
                control->cmsg_len < CMSG_LEN(sizeof(int))) continue;
            int received = -1;
            std::memcpy(&received, CMSG_DATA(control), sizeof(received));
            if (mirror >= 0) ::close(mirror);
            mirror = received;
        }
    }

    bool parse_frames(double now)
    {
        std::size_t at = 0u;
        while (at < incoming.size()) {
            if (incoming[at] == attach_mirror) {
                ++at;
                if (mirror < 0) {
                    incoming.erase(incoming.begin(),
                                   incoming.begin() + static_cast<std::ptrdiff_t>(at));
                    lost(now, "The mirror frame was refused because it had no descriptor. A retry starts in 2 seconds.");
                    return false;
                }
                if (state != State::connected) {
                    state = State::connected;
                    outage_stated = false;
                    results.emplace_back("The connection is ready.");
                }
                continue;
            }
            if (incoming[at] != attach_reason) {
                lost(now, "The socket sent an unknown frame. A retry starts in 2 seconds.");
                return false;
            }
            if (incoming.size() - at < 5u) break;
            const std::uint32_t length = static_cast<std::uint32_t>(incoming[at + 1u]) |
                (static_cast<std::uint32_t>(incoming[at + 2u]) << 8u) |
                (static_cast<std::uint32_t>(incoming[at + 3u]) << 16u) |
                (static_cast<std::uint32_t>(incoming[at + 4u]) << 24u);
            if (length > line_bound) {
                lost(now, "The socket reason is too long. A retry starts in 2 seconds.");
                return false;
            }
            if (incoming.size() - at < 5u + length) break;
            results.emplace_back(reinterpret_cast<const char *>(incoming.data() + at + 5u),
                                 length);
            at += 5u + length;
        }
        incoming.erase(incoming.begin(), incoming.begin() + static_cast<std::ptrdiff_t>(at));
        return true;
    }

    void receive(double now)
    {
        std::array<unsigned char, 4096> bytes{};
        std::array<char, CMSG_SPACE(sizeof(int))> controls{};
        while (fd >= 0) {
            iovec io{bytes.data(), bytes.size()};
            msghdr message{};
            message.msg_iov = &io;
            message.msg_iovlen = 1u;
            message.msg_control = controls.data();
            message.msg_controllen = controls.size();
            const ssize_t count = ::recvmsg(fd, &message, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
            if (count > 0) {
                take_descriptor(message);
                if (incoming.size() + static_cast<std::size_t>(count) > incoming_bound) {
                    lost(now, "The socket input overflow was refused. A retry starts in 2 seconds.");
                    return;
                }
                incoming.insert(incoming.end(), bytes.begin(), bytes.begin() + count);
                if (!parse_frames(now)) return;
                continue;
            }
            if (count == 0) {
                lost(now, "The connection was lost. A retry starts in 2 seconds.");
                return;
            }
            if (errno == EINTR) continue;
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                lost(now, "The connection was lost. A retry starts in 2 seconds.");
            }
            return;
        }
    }

    void send(double now)
    {
        while (fd >= 0 && sent < outgoing.size()) {
            const ssize_t count = ::send(fd, outgoing.data() + sent, outgoing.size() - sent,
                                         MSG_DONTWAIT | MSG_NOSIGNAL);
            if (count > 0) {
                sent += static_cast<std::size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
            lost(now, "The connection was lost. A retry starts in 2 seconds.");
            return;
        }
        if (sent == outgoing.size()) {
            outgoing.clear();
            sent = 0u;
        }
    }
};

Client::Client(std::filesystem::path journal)
    : impl_(std::make_unique<Impl>(std::move(journal))) {}

Client::~Client() = default;

void Client::tick(double now)
{
    if (impl_->fd < 0) {
        if (now >= impl_->retry_at) impl_->connect_now(now);
        return;
    }
    impl_->finish_connect(now);
    if (impl_->fd < 0) return;
    impl_->receive(now);
    if (impl_->fd >= 0) impl_->send(now);
}

bool Client::send_line(const std::string &line)
{
    if (line.size() > line_bound) {
        impl_->results.emplace_back("The line is longer than the input bound.");
        return false;
    }
    if (impl_->state != Impl::State::connected) {
        impl_->results.emplace_back("The line was refused because the connection is not ready.");
        return false;
    }
    const std::vector<unsigned char> frame = line_frame(line);
    impl_->outgoing.insert(impl_->outgoing.end(), frame.begin(), frame.end());
    return true;
}

std::vector<std::string> Client::take_results()
{
    std::vector<std::string> out;
    out.swap(impl_->results);
    return out;
}

const char *Client::connection() const
{
    switch (impl_->state) {
    case Impl::State::connected: return "connected";
    case Impl::State::attaching: return "attaching";
    case Impl::State::disconnected: return "not connected";
    }
    return "not connected";
}

int Client::mirror_descriptor() const { return impl_->mirror; }

bool verify_frame()
{
    const std::vector<unsigned char> frame = line_frame("say ready");
    return frame.size() == 14u && frame[0] == 'L' && frame[1] == 9u && frame[2] == 0u &&
           frame[3] == 0u && frame[4] == 0u &&
           std::string(frame.begin() + 5, frame.end()) == "say ready";
}

bool verify_outage()
{
    Client probe("/nonexistent/outage-check");
    probe.tick(0.0);
    probe.tick(3.0);
    probe.tick(6.0);
    const std::vector<std::string> first = probe.take_results();
    probe.tick(9.0);
    return first.size() == 1u && probe.take_results().empty();
}

} // namespace aotx::ctrl::client
