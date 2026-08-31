// Purpose: Show short results and refusals in one notification lane.
// Owns: The active notification list and dismissal times.
// Launch shape: One user interface thread adds and draws notifications.
// Lifetime: Notifications remain until their time limit expires.
#ifndef AOTX_CTRL_TOAST_HPP
#define AOTX_CTRL_TOAST_HPP

#include <string>
#include <vector>

namespace aotx::ctrl::toast {

enum class Severity { info, success, warning, error };

struct Notice {
    std::string text;
    Severity severity;
    double expires_at;
};

class Lane {
  public:
    void add(std::string text, Severity severity, double now, double seconds = 4.0);
    void draw(double now);

  private:
    std::vector<Notice> notices_;
};

} // namespace aotx::ctrl::toast

#endif
