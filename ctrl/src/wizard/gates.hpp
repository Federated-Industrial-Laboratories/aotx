// Purpose: Define the lifecycle facts and action latch for first-run page gates.
// Owns: Six page gate rules and repeated-action suppression.
// Launch shape: One interface or check evaluates one page at a time.
// Lifetime: Facts and latches remain valid for one first-run sequence.
#ifndef AOTX_CTRL_WIZARD_GATES_HPP
#define AOTX_CTRL_WIZARD_GATES_HPP

#include "instances/lifecycle.hpp"

#include <string>

namespace aotx::ctrl::wizard {

enum class Page { detect, build, model, activate, start, first_say };

struct GateFacts {
    bool detected = false;
    bool build_ready = false;
    bool catalog_read = false;
    bool model_on_disk = false;
    bool model_active = false;
    std::string phase;
    std::string connection;
    bool reply_received = false;
};

class ActionLatch {
  public:
    bool begin();
    void complete(bool accepted);
    void reset();
    bool active() const;
    bool completed() const;
    bool failed() const;

  private:
    bool active_ = false;
    bool attempted_ = false;
    bool completed_ = false;
};

bool gate_open(Page page, const GateFacts &facts);
bool start_gate_open(const instances::LiveInstance &instance);
bool verify_gates();

} // namespace aotx::ctrl::wizard

#endif
