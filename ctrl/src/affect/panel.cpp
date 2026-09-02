// Purpose: Draw affect and quality figures from the disk replica.
// Owns: No data outside the trace panel state.
// Launch shape: One interface thread draws the selected agent rows.
// Lifetime: Each frame reads the last 32 held turns.
#include "affect/panel.hpp"

#include "imgui.h"
#include "replica/replica.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace aotx::ctrl::affect {
namespace {

struct TurnRow {
    std::uint64_t turn = 0u;
    const replica::AffectTrace *affect = nullptr;
    const replica::QualityLine *quality = nullptr;
};

std::vector<unsigned> stream_agents(const replica::State &state)
{
    std::set<unsigned> held;
    for (const replica::AffectTrace &row : state.affect_traces()) held.insert(row.agent);
    for (const replica::QualityLine &row : state.quality_lines()) held.insert(row.agent);
    return {held.begin(), held.end()};
}

std::vector<TurnRow> joined_rows(const replica::State &state, unsigned agent)
{
    std::map<std::uint64_t, TurnRow> joined;
    for (const replica::AffectTrace &row : state.affect_traces()) {
        if (row.agent != agent) continue;
        TurnRow &held = joined[row.turn];
        held.turn = row.turn;
        held.affect = &row;
    }
    for (const replica::QualityLine &row : state.quality_lines()) {
        if (row.agent != agent) continue;
        TurnRow &held = joined[row.turn];
        held.turn = row.turn;
        held.quality = &row;
    }
    while (joined.size() > 32u) joined.erase(joined.begin());
    std::vector<TurnRow> rows;
    rows.reserve(joined.size());
    for (const auto &item : joined) rows.push_back(item.second);
    return rows;
}

std::string events(const replica::AffectTrace &row)
{
    std::string text;
    for (const std::string &word : row.reason) {
        if (!text.empty()) text += ", ";
        text += word;
    }
    return text.empty() ? "-" : text;
}

std::string state_values(const replica::AffectTrace &row)
{
    std::array<char, 128> text{};
    std::snprintf(text.data(), text.size(), "%.3f %.3f %.3f %.3f",
                  row.effective[0], row.effective[1], row.effective[2], row.effective[3]);
    return text.data();
}

std::string coherence(const replica::QualityLine &row)
{
    std::array<char, 64> text{};
    const std::string prompt = row.coherence_prompt.has_value()
        ? (std::snprintf(text.data(), text.size(), "%.3f", *row.coherence_prompt), text.data())
        : "-";
    const std::string turn = row.coherence_turn.has_value()
        ? (std::snprintf(text.data(), text.size(), "%.3f", *row.coherence_turn), text.data())
        : "-";
    return prompt + " / " + turn;
}

void text_cell(bool enabled, const std::string &text)
{
    ImGui::TableNextColumn();
    ImGui::BeginDisabled(!enabled);
    ImGui::TextUnformatted(text.c_str());
    ImGui::EndDisabled();
}

void figure_cell(bool enabled, const double *value)
{
    ImGui::TableNextColumn();
    ImGui::BeginDisabled(!enabled);
    if (value == nullptr) ImGui::TextUnformatted("-");
    else ImGui::Text("%.3f", *value);
    ImGui::EndDisabled();
}

void header_cell(bool enabled, const char *label)
{
    ImGui::TableNextColumn();
    ImGui::BeginDisabled(!enabled);
    ImGui::TableHeader(label);
    ImGui::EndDisabled();
}

void draw_table(const std::vector<TurnRow> &rows, bool affect_on, bool quality_on)
{
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;
    if (!ImGui::BeginTable("trace rows", 13, flags, ImVec2(0.0f, 300.0f))) return;
    ImGui::TableSetupScrollFreeze(1, 1);
    static const std::array<const char *, 13> columns = {
        "Turn", "Prompt valence", "Prompt arousal", "Reply valence",
        "Reply arousal", "Sycophancy", "Refusal", "Events", "Logprob",
        "Entropy", "Effective", "Coherence", "Repetition"};
    for (const char *column : columns) ImGui::TableSetupColumn(column);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    header_cell(true, columns[0]);
    for (std::size_t index = 1u; index <= 10u; ++index) {
        header_cell(affect_on, columns[index]);
    }
    header_cell(quality_on, columns[11]);
    header_cell(quality_on, columns[12]);
    for (const TurnRow &row : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(row.turn));
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->prompt[0]);
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->prompt[1]);
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->reply[0]);
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->reply[1]);
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->guard[0]);
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->guard[1]);
        text_cell(affect_on, row.affect == nullptr ? "-" : events(*row.affect));
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->logprob);
        figure_cell(affect_on, row.affect == nullptr ? nullptr : &row.affect->entropy);
        text_cell(affect_on, row.affect == nullptr ? "-" : state_values(*row.affect));
        text_cell(quality_on, row.quality == nullptr ? "-" : coherence(*row.quality));
        figure_cell(quality_on, row.quality == nullptr ? nullptr : &row.quality->repetition);
    }
    ImGui::EndTable();
}

void draw_explanations()
{
    ImGui::TextUnformatted("Prompt and reply values show valence and arousal readouts.");
    ImGui::TextUnformatted("Guard values show sycophancy and refusal readouts.");
    ImGui::TextUnformatted("Events show the conditions recorded for the turn.");
    ImGui::TextUnformatted("Logprob and entropy are turn means.");
    ImGui::TextUnformatted("Effective shows the four applied state values.");
    ImGui::TextUnformatted("Coherence shows prompt and prior-turn figures.");
    ImGui::TextUnformatted("Repetition shows the repeated token-trigram share.");
}

void draw_plots(const std::vector<TurnRow> &rows)
{
    std::array<std::vector<float>, 8> values;
    for (const TurnRow &row : rows) {
        if (row.affect == nullptr) continue;
        values[0].push_back(static_cast<float>(row.affect->prompt[0]));
        values[1].push_back(static_cast<float>(row.affect->prompt[1]));
        values[2].push_back(static_cast<float>(row.affect->reply[0]));
        values[3].push_back(static_cast<float>(row.affect->reply[1]));
        for (std::size_t axis = 0u; axis < 4u; ++axis) {
            values[4u + axis].push_back(static_cast<float>(row.affect->effective[axis]));
        }
    }
    if (values[0].empty()) return;
    static const std::array<const char *, 8> labels = {
        "Prompt valence", "Prompt arousal", "Reply valence", "Reply arousal",
        "Effective 0", "Effective 1", "Effective 2", "Effective 3"};
    ImGui::SeparatorText("Turn figures");
    for (std::size_t index = 0u; index < values.size(); ++index) {
        ImGui::PlotLines(labels[index], values[index].data(),
                         static_cast<int>(values[index].size()), 0, nullptr,
                         FLT_MAX, FLT_MAX, ImVec2(0.0f, 55.0f));
    }
}

} // namespace

void draw(State &view, const replica::State &state, bool *open)
{
    if (!ImGui::Begin("Trace", open)) {
        ImGui::End();
        return;
    }
    const bool affect_on = !state.affect_traces().empty();
    const bool quality_on = !state.quality_lines().empty();
    if (!affect_on && !quality_on) {
        ImGui::TextDisabled("The affect substrate and the quality stream are off. Set affect.on or quality.on to 1 to start one.");
        ImGui::End();
        return;
    }
    const std::vector<unsigned> agents = stream_agents(state);
    if (std::find(agents.begin(), agents.end(), view.agent) == agents.end()) {
        view.agent = agents.front();
    }
    const std::string selected = "Agent " + std::to_string(view.agent);
    if (ImGui::BeginCombo("Agent", selected.c_str())) {
        for (const unsigned agent : agents) {
            const std::string label = "Agent " + std::to_string(agent);
            if (ImGui::Selectable(label.c_str(), agent == view.agent)) view.agent = agent;
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Select an agent from the streams.");
    if (!affect_on) {
        ImGui::TextDisabled("The affect substrate is off. Set affect.on to 1 to start it.");
    }
    if (!quality_on) {
        ImGui::TextDisabled("The quality stream is off. Set quality.on to 1 to start it.");
    }
    const std::vector<TurnRow> rows = joined_rows(state, view.agent);
    const float available = ImGui::GetContentRegionAvail().x;
    const float table_width = std::max(560.0f, available - 420.0f);
    ImGui::BeginChild("trace table area", ImVec2(table_width, 320.0f), false);
    draw_table(rows, affect_on, quality_on);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginGroup();
    draw_explanations();
    ImGui::EndGroup();
    if (affect_on) draw_plots(rows);
    ImGui::End();
}

} // namespace aotx::ctrl::affect
