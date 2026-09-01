// Purpose: Draw simulated tick, ring, memory, and agent metrics.
// Owns: Palette cues for metric state.
// Launch shape: One panel draws all current monitor values.
// Lifetime: Values update from the simulated state each frame.
#include "monitor/monitor.hpp"

#include "client/client.hpp"
#include "imgui.h"
#include "replica/replica.hpp"
#include "theme/theme.hpp"

#include <algorithm>
#include <set>

namespace aotx::ctrl::monitor {
namespace {

ImVec4 state_color(const std::string &state)
{
    if (state == "run" || state == "idle") return theme::palette().running;
    if (state == "prompt" || state == "tool") return theme::palette().attaching;
    return theme::palette().stopped;
}

const char *agent_state(std::uint64_t state)
{
    static const char *names[] = {"free", "idle", "prompt", "run", "tool", "post"};
    return state < 6u ? names[state] : "unknown";
}

} // namespace

void draw(const sim::State &state, bool *open)
{
    if (!ImGui::Begin("Monitor", open)) {
        ImGui::End();
        return;
    }
    const ImVec4 tick_color = state.tick_rate_hz >= 95.0f
                                  ? theme::palette().running
                                  : theme::palette().severity_warning;
    ImGui::TextColored(tick_color, "Tick rate %.1f Hz", state.tick_rate_hz);
    const ImVec4 ring_color = state.ring_occupancy < 0.75f
                                  ? theme::palette().running
                                  : theme::palette().severity_error;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ring_color);
    ImGui::ProgressBar(state.ring_occupancy, ImVec2(-1.0f, 0.0f), "Ring occupancy");
    ImGui::PopStyleColor();
    ImGui::SeparatorText("Card memory");
    for (const sim::Instance &instance : state.instances) {
        for (const sim::Card &card : instance.cards) {
            ImGui::Text("%s / %s", instance.name.c_str(), card.name.c_str());
            const float use = card.memory_total_mib == 0
                                  ? 0.0f
                                  : static_cast<float>(card.memory_used_mib) /
                                        static_cast<float>(card.memory_total_mib);
            const ImVec4 color = use < 0.8f ? theme::palette().running
                                            : theme::palette().severity_error;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
            ImGui::ProgressBar(use, ImVec2(-1.0f, 0.0f));
            ImGui::PopStyleColor();
            ImGui::TextDisabled("%u MiB of %u MiB", card.memory_used_mib,
                                card.memory_total_mib);
        }
    }
    ImGui::SeparatorText("Agents");
    for (const sim::Agent &agent : state.agents) {
        ImGui::TextColored(state_color(agent.state), "%s  %s  %s  %u pages",
                           agent.name.c_str(), agent.role.c_str(), agent.state.c_str(), agent.pages);
    }
    ImGui::End();
}

void draw(Telemetry &telemetry, const replica::State &state, const client::Client &client,
          double now, bool *open)
{
    telemetry.tick(client.mirror_descriptor(), now);
    if (!ImGui::Begin("Monitor", open)) {
        ImGui::End();
        return;
    }
    const MirrorSample &sample = telemetry.mirror();
    const ImVec4 mirror_color = sample.available ? theme::palette().running
                                                  : theme::palette().severity_warning;
    ImGui::TextColored(mirror_color, "Tick %llu",
                       static_cast<unsigned long long>(sample.tick));
    ImGui::TextColored(mirror_color, "Tick rate %.1f Hz", sample.tick_rate);
    ImGui::TextColored(mirror_color, "%s", sample.result.c_str());
    const float ring = ring_fraction(sample.ring_used, sample.ring_slots);
    if (ring >= 0.0f) {
        const ImVec4 live_ring_color = ring < 0.75f ? theme::palette().running
                                                    : theme::palette().severity_error;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, live_ring_color);
        ImGui::ProgressBar(ring, ImVec2(-1.0f, 0.0f), "Ring occupancy");
        ImGui::PopStyleColor();
        ImGui::Text("%llu of %llu device ring slots hold records.",
                    static_cast<unsigned long long>(sample.ring_used),
                    static_cast<unsigned long long>(sample.ring_slots));
    } else {
        ImGui::TextColored(theme::palette().severity_info,
                           "The system does not publish the ring occupancy.");
    }
    ImGui::Text("Token rate %.1f tokens/s", state.token_rate());
    unsigned resident = 0u;
    for (const replica::PageStat &page : state.pages()) resident += page.residency;
    const float occupancy = state.pages().empty() ? 0.0f :
        static_cast<float>(resident) / static_cast<float>(state.pages().size());
    ImGui::ProgressBar(occupancy, ImVec2(-1.0f, 0.0f), "Page occupancy");
    ImGui::Text("%u resident pages of %zu measured pages.", resident, state.pages().size());

    ImGui::SeparatorText("Card memory");
    const ImVec4 card_result_color = telemetry.cards().empty()
                                         ? theme::palette().severity_warning
                                         : theme::palette().running;
    ImGui::TextColored(card_result_color, "%s", telemetry.card_result().c_str());
    for (const CardMemory &card : telemetry.cards()) {
        const float use = card.total_mib == 0u ? 0.0f :
            static_cast<float>(static_cast<double>(card.used_mib) /
                               static_cast<double>(card.total_mib));
        const ImVec4 color = use < 0.8f ? theme::palette().running
                                        : theme::palette().severity_error;
        ImGui::TextColored(color, "Card %u: %s", card.index, card.name.c_str());
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
        ImGui::ProgressBar(use, ImVec2(-1.0f, 0.0f));
        ImGui::PopStyleColor();
        ImGui::TextColored(color, "%llu MiB of %llu MiB",
                           static_cast<unsigned long long>(card.used_mib),
                           static_cast<unsigned long long>(card.total_mib));
    }

    ImGui::SeparatorText("Agents");
    if (state.agent_states().empty()) {
        ImGui::TextColored(theme::palette().stopped, "No derived agent state is available.");
    }
    for (const replica::AgentState &agent : state.agent_states()) {
        const char *name = agent_state(agent.state);
        ImGui::TextColored(state_color(name), "Agent %llu  %s  role %llu  turn %llu",
                           static_cast<unsigned long long>(agent.agent), name,
                           static_cast<unsigned long long>(agent.role),
                           static_cast<unsigned long long>(agent.turn));
    }
    ImGui::SeparatorText("Page map");
    ImGui::TextDisabled("The page statistics flush every 64 ticks. This panel does not control pages.");
    std::set<unsigned> page_agents;
    for (const replica::PageStat &page : state.pages()) page_agents.insert(page.agent);
    if (page_agents.empty()) ImGui::TextDisabled("No page measurement is available.");
    for (const unsigned agent : page_agents) {
        ImGui::PushID(static_cast<int>(agent));
        ImGui::Text("Agent %u", agent);
        double largest = 0.0;
        for (const replica::PageStat &page : state.pages()) {
            if (page.agent == agent) largest = std::max(largest, page.mass);
        }
        for (const replica::PageStat &page : state.pages()) {
            if (page.agent != agent) continue;
            const float amount = largest == 0.0 ? 0.0f :
                static_cast<float>(page.mass / largest);
            const ImVec4 color = page.residency != 0u ? theme::palette().running
                                                       : theme::palette().stopped;
            ImGui::PushID(static_cast<int>(page.page));
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
            const std::string label = "Page " + std::to_string(page.page) +
                (page.residency != 0u ? " resident" : " not resident") +
                ", mass " + std::to_string(page.mass);
            ImGui::ProgressBar(amount, ImVec2(-1.0f, 0.0f), label.c_str());
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
        ImGui::PopID();
    }
    ImGui::End();
}

} // namespace aotx::ctrl::monitor
