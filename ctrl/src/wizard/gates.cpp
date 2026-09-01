// Purpose: Evaluate first-run page gates from shared system state.
// Owns: Gate truth tables and action latch transitions.
// Launch shape: One caller evaluates one page or one startup fixture.
// Lifetime: No state exists outside the action latch given by the caller.
#include "wizard/gates.hpp"

namespace aotx::ctrl::wizard {

bool ActionLatch::begin()
{
    if (active_ || attempted_) return false;
    active_ = true;
    attempted_ = true;
    return true;
}

void ActionLatch::complete(bool accepted)
{
    active_ = false;
    completed_ = accepted;
}

void ActionLatch::reset()
{
    active_ = false;
    attempted_ = false;
    completed_ = false;
}

bool ActionLatch::active() const { return active_; }
bool ActionLatch::completed() const { return completed_; }
bool ActionLatch::failed() const { return attempted_ && !active_ && !completed_; }

bool gate_open(Page page, const GateFacts &facts)
{
    switch (page) {
    case Page::detect: return facts.detected;
    case Page::build: return facts.build_ready;
    case Page::model: return facts.catalog_read && facts.model_on_disk;
    case Page::activate: return facts.catalog_read && facts.model_active;
    case Page::start:
        return facts.phase == "running" && facts.connection == "connected";
    case Page::first_say: return facts.reply_received;
    }
    return false;
}

bool start_gate_open(const instances::LiveInstance &instance)
{
    GateFacts facts;
    facts.phase = instance.phase;
    facts.connection = instance.connection;
    return instance.state == instances::LiveState::running && gate_open(Page::start, facts);
}

bool verify_gates()
{
    GateFacts facts;
    ActionLatch action;
    if (gate_open(Page::detect, facts) || !action.begin() || action.begin()) return false;
    action.complete(true);
    if (!action.completed() || action.begin()) return false;
    action.reset();
    if (!action.begin()) return false;
    action.complete(false);
    if (action.active() || action.completed() || !action.failed() || action.begin()) {
        return false;
    }
    action.reset();
    if (!action.begin()) return false;

    facts.detected = true;
    if (!gate_open(Page::detect, facts) || gate_open(Page::build, facts)) return false;
    facts.build_ready = true;
    if (!gate_open(Page::build, facts) || gate_open(Page::model, facts)) return false;
    facts.catalog_read = true;
    if (gate_open(Page::model, facts) || gate_open(Page::activate, facts)) return false;
    facts.model_on_disk = true;
    if (!gate_open(Page::model, facts) || gate_open(Page::activate, facts)) return false;
    facts.model_active = true;
    if (!gate_open(Page::activate, facts) || gate_open(Page::start, facts)) return false;
    facts.phase = "placing";
    facts.connection = "attaching";
    if (gate_open(Page::start, facts)) return false;
    facts.phase = "replaying";
    if (gate_open(Page::start, facts)) return false;
    facts.phase = "running";
    if (gate_open(Page::start, facts)) return false;
    facts.connection = "connected";
    if (!gate_open(Page::start, facts) || gate_open(Page::first_say, facts)) return false;
    instances::LiveInstance instance;
    instance.phase = facts.phase;
    instance.connection = facts.connection;
    if (start_gate_open(instance)) return false;
    instance.state = instances::LiveState::running;
    if (!start_gate_open(instance)) return false;
    instance.connection = "attaching";
    if (start_gate_open(instance)) return false;
    facts.reply_received = true;
    return gate_open(Page::first_say, facts);
}

} // namespace aotx::ctrl::wizard
