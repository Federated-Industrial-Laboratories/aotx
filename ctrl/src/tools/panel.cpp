// Purpose: Display reported tool selection and send named changes to the device.
// Owns: Local query status; the device owns every accepted selection.
// Launch shape: One interface thread draws the groups of one conversation.
// Lifetime: Choices persist in settings records after the device accepts them.
#include "tools/panel.hpp"
#include "cuda/tool/policy.h"
#include "imgui.h"
#include <algorithm>

namespace aotx::ctrl::tools {
namespace {
void send(View &view, client::Client &client, const std::string &command)
{
    view.result = client.send_line(command) ? "The change was sent. The next turn uses the accepted choice."
                                          : "The command was not sent. Check the instance connection.";
}
}

void draw(View &view, const replica::State &state, client::Client &client, unsigned agent)
{
    if (!ImGui::CollapsingHeader("System tools")) return;
    const std::string boot = state.boots().empty() ? "" : state.boots().front().name;
    const std::string identity = state.journal().string() + ":" + boot + ":" + std::to_string(agent);
    const std::string prefix = "agent " + std::to_string(agent) + " tools";
    bool refresh = ImGui::Button("Refresh##tools");
    request_status(view.query, client, identity, prefix, refresh);
    ImGui::SameLine();
    ImGui::TextDisabled("Read the next turn selection from the device.");
    const auto found = std::find_if(state.tool_policies().begin(), state.tool_policies().end(),
        [agent](const replica::ToolPolicy &policy) { return policy.agent == agent; });
    if (!view.query.result.empty()) ImGui::TextWrapped("%s", view.query.result.c_str());
    if (found == state.tool_policies().end()) {
        ImGui::TextDisabled("The device has not reported this conversation's tool choices.");
        return;
    }
    const replica::ToolPolicy &policy = *found;
    const char *names[] = {
#define AOTX_TOOL_NAME(name) name,
        AOTX_TOOL_POLICY_NAMES(AOTX_TOOL_NAME)
#undef AOTX_TOOL_NAME
    };
    ImGui::TextWrapped("Conversation choices override instance defaults. Role grants and model availability still apply.");
    ImGui::TextWrapped("Changes apply at the next turn. History and internal conversation memory remain active.");
    if (ImGui::Button("Instance off##tools")) send(view, client, "tool all off");
    ImGui::SameLine();
    if (ImGui::Button("Instance on##tools")) send(view, client, "tool all on");
    if (ImGui::Button("Inherit all##tools")) send(view, client, prefix + " all inherit");
    ImGui::SameLine();
    if (ImGui::Button("Conversation off##tools")) send(view, client, prefix + " all off");
    ImGui::SameLine();
    if (ImGui::Button("Conversation on##tools")) send(view, client, prefix + " all on");
    if (ImGui::BeginTable("tool-choices", 4, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Tool"); ImGui::TableSetupColumn("Instance");
        ImGui::TableSetupColumn("Conversation"); ImGui::TableSetupColumn("Effective");
        ImGui::TableHeadersRow();
        for (unsigned group = 0u; group < AOTX_TOOL_POLICY_GROUPS; ++group) {
            ImGui::PushID(static_cast<int>(group));
            const unsigned bit = 1u << group;
            ImGui::TableNextRow(); ImGui::TableNextColumn();
            ImGui::TextUnformatted(names[group]);
            ImGui::TableNextColumn();
            bool enabled = (policy.defaults & bit) != 0u;
            if (ImGui::Checkbox("On##instance", &enabled))
                send(view, client, std::string("tool ") + names[group] + (enabled ? " on" : " off"));
            ImGui::TableNextColumn();
            int choice = static_cast<int>((policy.choices >> (2u * group)) & 3u);
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::Combo("##choice", &choice, "Inherit\0Off\0On\0")) {
                const char *word[] = {"inherit", "off", "on"};
                send(view, client, prefix + " " + names[group] + " " + word[choice]);
            }
            ImGui::TableNextColumn();
            const char *effective = (policy.effective & bit) != 0u ? "On"
                : (policy.selected & bit) != 0u ? "Unavailable" : "Off";
            ImGui::TextUnformatted(effective);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Imported tools share one group. On means at least one granted tool is available.");
    if (!view.result.empty()) ImGui::TextWrapped("%s", view.result.c_str());
}
}
