// Purpose: Define the attach socket client for one live system.
// Owns: The socket, the mirror descriptor, and queued line frames.
// Launch shape: One interface thread polls one connection.
// Lifetime: A client exists while one journal is selected.
#ifndef AOTX_CTRL_CLIENT_HPP
#define AOTX_CTRL_CLIENT_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aotx::ctrl::client {

class Client {
  public:
    explicit Client(std::filesystem::path journal);
    ~Client();
    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;

    void tick(double now);
    bool send_line(const std::string &line);
    std::vector<std::string> take_results();
    const char *connection() const;
    int mirror_descriptor() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool verify_frame();

} // namespace aotx::ctrl::client

#endif
