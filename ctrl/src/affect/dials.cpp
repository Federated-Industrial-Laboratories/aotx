// Purpose: Draw bounded controls with calibration and trace figures.
// Owns: The Dials window and edits sent through the settings surface.
// Launch shape: One interface frame draws thirteen setting rows.
// Lifetime: The view keeps edits until its instance binding changes.
#include "affect/dials.hpp"

#include "client/client.hpp"
#include "imgui.h"
#include "replica/replica.hpp"
#include "settings/settings.hpp"
#include "toast/toast.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace aotx::ctrl::affect::dials {
namespace {

enum class Figure { switch_value, probe, state, valence, arousal, steer,
                    entropy_shift, class_shift, budget_spent };

struct Spec {
    const char *key;
    float initial;
    float least;
    float most;
    const char *sentence;
    Figure figure;
    bool switch_control;
};

constexpr std::array<Spec, 13> specs = {{
    {"affect.on", 0.0f, 0.0f, 1.0f,
     "Start the affect substrate at the next sequence.", Figure::switch_value, true},
    {"quality.on", 0.0f, 0.0f, 1.0f,
     "Start the quality stream at the next sequence.", Figure::switch_value, true},
    {"affect.probe_gain", 0.0f, 0.0f, 1.0f,
     "Set the probe contribution to the state update.", Figure::probe, false},
    {"affect.decay_fast", 0.5f, 0.0f, 0.99f,
     "Set the fast-state retention for each turn.", Figure::state, false},
    {"affect.decay_slow", 0.9f, 0.0f, 0.99f,
     "Set the slow-state retention for each turn.", Figure::state, false},
    {"affect.gain_fast", 0.5f, 0.0f, 2.0f,
     "Set the fast-state response to each event.", Figure::state, false},
    {"affect.gain_slow", 0.1f, 0.0f, 2.0f,
     "Set the slow-state response to each event.", Figure::state, false},
    {"affect.cap_valence", 1.0f, 0.0f, 1.0f,
     "Limit the effective valence magnitude.", Figure::valence, false},
    {"affect.cap_arousal", 1.0f, 0.0f, 1.0f,
     "Limit the effective arousal magnitude.", Figure::arousal, false},
    {"affect.temperature_gain", 0.0f, -1.0f, 1.0f,
     "Set the arousal coupling to sampler temperature.", Figure::entropy_shift, false},
    {"affect.voice_gain", 0.0f, -1.0f, 1.0f,
     "Set the valence coupling to the sampler voice bias.", Figure::class_shift, false},
    {"affect.steer_gain", 0.0f, 0.0f, 1.0f,
     "Set the effective-state dose for composite steering.", Figure::steer, false},
    {"affect.budget", 0.25f, 0.0f, 4.0f,
     "Limit the composite steering budget in nats.", Figure::budget_spent, false},
}};

void initialize(State &view, const replica::State &state)
{
    if (view.initialized) return;
    for (std::size_t index = 0u; index < specs.size(); ++index) {
        std::string text;
        view.values[index] = specs[index].initial;
        if (!replica::setting_value(state.settings(), specs[index].key, text)) continue;
        char *end = nullptr;
        const double value = std::strtod(text.c_str(), &end);
        if (end == text.c_str() + text.size() && std::isfinite(value) &&
            value >= specs[index].least && value <= specs[index].most) {
            view.values[index] = static_cast<float>(value);
        }
    }
    view.initialized = true;
}

const replica::AffectTrace *last_trace(const replica::State &state, unsigned &agent)
{
    if (!state.agents().empty()) {
        const std::size_t selected = std::min(state.selected_agent(), state.agents().size() - 1u);
        agent = state.agents()[selected].id;
    } else if (!state.affect_traces().empty()) {
        agent = state.affect_traces().back().agent;
    }
    for (auto row = state.affect_traces().rbegin(); row != state.affect_traces().rend(); ++row) {
        if (row->agent == agent) return &*row;
    }
    return nullptr;
}

std::string probe_figure(const replica::State &state)
{
    if (state.probe_accuracies().empty()) return {};
    std::array<char, 256> text{};
    std::size_t used = 0u;
    for (const replica::ProbeAccuracy &probe : state.probe_accuracies()) {
        const int made = std::snprintf(text.data() + used, text.size() - used, "%s%s %.3f",
            used == 0u ? "Accuracy:" : ",", probe.name.c_str(), probe.accuracy);
        if (made < 0 || static_cast<std::size_t>(made) >= text.size() - used) break;
        used += static_cast<std::size_t>(made);
    }
    return text.data();
}

std::string figure_text(Figure figure, float value, const replica::State &state,
                        const replica::AffectTrace *trace)
{
    std::array<char, 256> text{};
    if (figure == Figure::switch_value) {
        std::snprintf(text.data(), text.size(), "The file value is %u.", value >= 0.5f ? 1u : 0u);
    } else if (figure == Figure::probe) {
        return probe_figure(state);
    } else if ((figure == Figure::state || figure == Figure::valence ||
                figure == Figure::arousal) && trace != nullptr) {
        if (figure == Figure::state) {
            std::snprintf(text.data(), text.size(), "Effective valence %.3f; arousal %.3f.",
                          trace->effective[0], trace->effective[1]);
        } else {
            const std::size_t axis = figure == Figure::valence ? 0u : 1u;
            std::snprintf(text.data(), text.size(), "The last effective value is %.3f.",
                          trace->effective[axis]);
        }
    } else if (figure == Figure::steer && state.calibration().has_value()) {
        const replica::Calibration &row = *state.calibration();
        std::snprintf(text.data(), text.size(),
            "K diagonal %.4f, %.4f; dose ratios %.3f, %.3f.",
            row.K[0][0], row.K[1][1], row.ratio[0], row.ratio[1]);
    } else if (figure == Figure::entropy_shift && trace != nullptr &&
               trace->entropy_shift.has_value()) {
        std::snprintf(text.data(), text.size(),
                      "The entropy shift of the last turn is %.4f nats.",
                      *trace->entropy_shift);
    } else if (figure == Figure::class_shift && trace != nullptr &&
               trace->class_shift.has_value()) {
        std::snprintf(text.data(), text.size(),
                      "The class shift of the last turn is %.4f.", *trace->class_shift);
    } else if (figure == Figure::budget_spent && trace != nullptr &&
               trace->budget_spent.has_value()) {
        std::snprintf(text.data(), text.size(),
                      "The budget spent of the last turn is %.4f nats.",
                      *trace->budget_spent);
    }
    return text.data();
}

std::string value_text(const Spec &spec, float value)
{
    if (spec.switch_control) return value >= 0.5f ? "1" : "0";
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.4f", value);
    std::string made = text.data();
    while (!made.empty() && made.back() == '0') made.pop_back();
    if (!made.empty() && made.back() == '.') made.pop_back();
    return made;
}

void control(const Spec &spec, float &value)
{
    if (spec.switch_control) {
        bool on = value >= 0.5f;
        if (ImGui::Checkbox("##value", &on)) value = on ? 1.0f : 0.0f;
    } else {
        ImGui::SetNextItemWidth(220.0f);
        ImGui::SliderFloat("##value", &value, spec.least, spec.most, "%.4f");
    }
}

} // namespace

void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open)
{
    initialize(view, state);
    if (!ImGui::Begin("Dials", open)) {
        ImGui::End();
        return;
    }
    const bool affect_on = view.values[0] >= 0.5f;
    const bool quality_on = view.values[1] >= 0.5f;
    unsigned agent = 0u;
    const replica::AffectTrace *trace = last_trace(state, agent);
    FigureAvailability figures;
    figures.calibration = state.calibration().has_value();
    figures.probes = !state.probe_accuracies().empty();
    figures.trace = trace != nullptr;
    figures.budget_spent = trace != nullptr && trace->budget_spent.has_value();
    figures.entropy_shift = trace != nullptr && trace->entropy_shift.has_value();
    figures.class_shift = trace != nullptr && trace->class_shift.has_value();
    ImGui::TextDisabled("%s", window_sentence(window_state(affect_on, quality_on, figures)));
    ImGui::Text("Figures are for agent %u.", agent);
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("Affect setting dials", 3, flags)) {
        ImGui::TableSetupColumn("Setting", ImGuiTableColumnFlags_WidthFixed, 190.0f);
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, 240.0f);
        ImGui::TableSetupColumn("Explanation and figure");
        ImGui::TableHeadersRow();
        for (std::size_t index = 0u; index < specs.size(); ++index) {
            const Spec &spec = specs[index];
            ImGui::PushID(static_cast<int>(index));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(spec.key);
            ImGui::TableSetColumnIndex(1);
            const Setting setting = static_cast<Setting>(index);
            ImGui::BeginDisabled(!control_enabled(setting, figures));
            control(spec, view.values[index]);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(spec.sentence);
            const std::string figure = figure_text(spec.figure, view.values[index], state, trace);
            if (figure.empty()) {
                ImGui::TextDisabled("No calibration figure is loaded for this control.");
            } else {
                ImGui::TextDisabled("%s", figure.c_str());
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (ImGui::Button("Apply")) {
        for (std::size_t index = 0u; index < specs.size(); ++index) {
            settings::apply_value(state, client, toasts, now, specs[index].key,
                                  value_text(specs[index], view.values[index]));
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Apply all dial values to the next sequence.");
    if (ImGui::Button("Reset")) {
        for (std::size_t index = 0u; index < specs.size(); ++index) {
            view.values[index] = specs[index].initial;
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Set all controls to their default values.");
    ImGui::End();
}

} // namespace aotx::ctrl::affect::dials
