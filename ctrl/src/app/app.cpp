// Purpose: Run GLFW, OpenGL, ImGui, and all control shell modules.
// Owns: Window resources, ImGui resources, and the frame loop.
// Launch shape: One process draws one frame at a time.
// Lifetime: Resources start in order and stop in reverse order.
#include "app/app.hpp"

#ifdef AOTX_AFFECT
#include "affect/dials.hpp"
#include "affect/panel.hpp"
#endif
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "browser/browser.hpp"
#include "chat/chat.hpp"
#include "chat/format.hpp"
#include "chat/persona.hpp"
#include "client/client.hpp"
#include "control/control.hpp"
#include "imgui.h"
#include "instances/instances.hpp"
#include "model/model.hpp"
#include "module/module.hpp"
#include "monitor/monitor.hpp"
#include "replica/json.hpp"
#include "replica/replica.hpp"
#include "settings/settings.hpp"
#include "shell/shell.hpp"
#include "sim/sim.hpp"
#include "sync/sync.hpp"
#include "theme/theme.hpp"
#include "toast/toast.hpp"
#include "voice/voice.hpp"
#include "voice/panel.hpp"
#include "wizard/wizard.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace aotx::ctrl::app {
namespace {

struct Options {
    int frame_limit = -1;
    bool simulated = false;
    bool help = false;
    std::filesystem::path journal;
    std::filesystem::path settings;
    std::filesystem::path build;
};

void glfw_error(int, const char *description)
{
    std::fprintf(stderr, "GLFW reports: %s\n", description);
}

std::filesystem::path config_directory()
{
    const char *xdg = std::getenv("XDG_CONFIG_HOME");
    const char *home = std::getenv("HOME");
    if (xdg != nullptr && xdg[0] != '\0') return std::filesystem::path(xdg) / "aotx";
    if (home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".config" / "aotx";
    }
    return {};
}

/* The last bound journal is kept, so a bare start needs no argument. */
std::filesystem::path remembered_journal()
{
    const std::filesystem::path base = config_directory();
    if (base.empty()) return {};
    std::ifstream file(base / "journal");
    std::string line;
    if (!std::getline(file, line) || line.empty()) return {};
    std::error_code error;
    if (!std::filesystem::is_directory(line, error)) return {};
    return line;
}

void remember_journal(const std::filesystem::path &journal)
{
    const std::filesystem::path base = config_directory();
    if (base.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(base, error);
    if (error) return;
    std::ofstream file(base / "journal", std::ios::trunc);
    std::error_code whole_error;
    const std::filesystem::path whole = std::filesystem::absolute(journal, whole_error);
    file << (whole_error ? journal : whole).lexically_normal().string() << '\n';
}

/* The initial journal keeps its saved name and card. */
void read_saved_identity(instances::Definition &definition)
{
    std::ifstream file(config_directory() / "instances.jsonl");
    std::string line;
    while (std::getline(file, line)) {
        replica::json::Value value;
        std::string journal;
        std::string name;
        unsigned long long card = 0u;
        if (!replica::json::parse(line, value) ||
            !replica::json::text(value, "journal", journal) ||
            std::filesystem::path(journal).lexically_normal() !=
                definition.journal.lexically_normal()) continue;
        if (replica::json::text(value, "name", name) && !name.empty()) definition.name = name;
        if (replica::json::number(value, "card", card) &&
            card <= std::numeric_limits<unsigned>::max()) definition.card = static_cast<unsigned>(card);
        return;
    }
}

/* A first bare start makes a home of its own under the user data directory. */
std::filesystem::path fresh_home_journal()
{
    const char *xdg = std::getenv("XDG_DATA_HOME");
    const char *home = std::getenv("HOME");
    std::filesystem::path base;
    if (xdg != nullptr && xdg[0] != '\0') {
        base = std::filesystem::path(xdg) / "aotx";
    } else if (home != nullptr && home[0] != '\0') {
        base = std::filesystem::path(home) / ".local/share/aotx";
    } else {
        return {};
    }
    std::error_code error;
    std::filesystem::create_directories(base / "journal", error);
    if (error) return {};
    std::filesystem::create_directories(base / "models", error);
    return base / "journal";
}

void print_usage()
{
    std::fputs("aotx_ctrl [--journal <dir>] [--settings <file>] [--sim] [--frames <n>] [--help]\n"
               "  --journal   the journal directory of the system to control\n"
               "  --settings  the settings file that names the journal directory\n"
               "  --sim       run against simulated data, with no system\n"
               "  --frames    stop after this count of frames\n"
               "  --help      print this text and stop\n"
               "With no journal option, the settings file names it, then the last journal\n"
               "opened, then a new journal in the user data directory.\n", stdout);
}

bool parse_options(int argc, char **argv, Options &options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            return true;
        }
        if (argument == "--sim") {
            options.simulated = true;
            continue;
        }
        if ((argument == "--journal" || argument == "--settings") && index + 1 < argc) {
            const std::filesystem::path value = argv[++index];
            if (argument == "--journal") options.journal = value;
            else options.settings = value;
            continue;
        }
        if (argument != "--frames" || index + 1 >= argc) {
            std::fputs("AOTX-CTRL refuses an unknown option.\n", stderr);
            print_usage();
            return false;
        }
        const std::string value = argv[++index];
        int parsed = -1;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc() || result.ptr != value.data() + value.size() || parsed < 0) {
            std::fputs("AOTX-CTRL refuses an invalid frame count.\n", stderr);
            return false;
        }
        options.frame_limit = parsed;
    }
    if (!options.simulated && options.journal.empty()) {
        std::string journal;
        if (!options.settings.empty() &&
            replica::setting_value(options.settings, "journal.dir", journal) &&
            !journal.empty()) {
            std::filesystem::path found = journal;
            if (found.is_relative()) {
                found = std::filesystem::path(options.settings).parent_path() / found;
            }
            options.journal = found;
        }
    }
    if (!options.simulated && options.journal.empty()) {
        options.journal = remembered_journal();
    }
    if (!options.simulated && options.journal.empty()) {
        options.journal = fresh_home_journal();
        if (options.journal.empty()) {
            std::fputs("AOTX-CTRL cannot create its data directory.\n", stderr);
            return false;
        }
    }
    return true;
}

bool make_layout_path(std::string &path)
{
    const char *xdg = std::getenv("XDG_CONFIG_HOME");
    const char *home = std::getenv("HOME");
    std::filesystem::path directory;
    if (xdg != nullptr && xdg[0] != '\0') {
        directory = xdg;
    } else if (home != nullptr && home[0] != '\0') {
        directory = std::filesystem::path(home) / ".config";
    } else {
        std::fputs("AOTX-CTRL cannot find the user configuration directory.\n", stderr);
        return false;
    }
    directory /= "aotx";
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        std::fputs("AOTX-CTRL cannot create the user configuration directory.\n", stderr);
        return false;
    }
    path = (directory / "ctrl-layout.ini").string();
    return true;
}

toast::Severity result_severity(const std::string &text)
{
    if (text.rfind("No running system", 0) == 0) return toast::Severity::warning;
    return text.find("refused") != std::string::npos ||
           text.find("failed") != std::string::npos ||
           text.find("does not") != std::string::npos ||
           text.find("cannot") != std::string::npos
        ? toast::Severity::error : toast::Severity::success;
}

int run_loop(GLFWwindow *window, const Options &options, const std::string &layout_path)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.IniFilename = layout_path.c_str();
    theme::apply();

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true) || !ImGui_ImplOpenGL3_Init("#version 130")) {
        std::fputs("AOTX-CTRL cannot start the graphics interface.\n", stderr);
        ImGui::DestroyContext();
        return 1;
    }

    sim::State simulated;
    instances::Lifecycle lifecycle;
    bool fresh = false;
    if (!options.simulated) {
        lifecycle.set_registry(config_directory() / "instances.jsonl");
        std::string models;
        std::string roles;
        std::string tools;
        const std::filesystem::path settings = options.settings.empty()
            ? options.journal.parent_path() / "aotx.settings" : options.settings;
        std::filesystem::path source = settings;
        if (!std::filesystem::is_regular_file(source)) {
            source = options.journal.parent_path().parent_path() / "aotx.settings";
        }
        if (std::filesystem::is_regular_file(source) &&
            replica::setting_value(source, "models.dir", models)) {
            std::filesystem::path found = models;
            if (found.is_relative()) found = source.parent_path() / found;
            models = found.string();
        } else {
            const std::filesystem::path near = options.journal.parent_path() / "models";
            const std::filesystem::path above =
                options.journal.parent_path().parent_path() / "models";
            models = (std::filesystem::is_directory(near) ||
                      !std::filesystem::is_directory(above) ? near : above).string();
        }
        if (std::filesystem::is_regular_file(source)) {
            replica::setting_value(source, "models.roles", roles);
            replica::setting_value(source, "tools.root", tools);
        }
        instances::Definition local;
        local.name = "Local instance";
        local.journal = options.journal;
        local.settings = settings;
        local.build = options.build;
        local.models = models;
        if (!roles.empty()) local.roles = roles;
        if (!tools.empty()) {
            std::filesystem::path root = tools;
            if (root.is_relative()) root = source.parent_path() / root;
            local.tools = root;
        }
        if (local.tools.empty()) {
            const char *user_home = std::getenv("HOME");
            if (user_home != nullptr && user_home[0] != '\0') local.tools = user_home;
        }
        read_saved_identity(local);
        if (!lifecycle.seed(std::move(local), true)) {
            std::fprintf(stderr, "AOTX-CTRL: %s\n", lifecycle.refusal().c_str());
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            return 2;
        }
        remember_journal(options.journal);
        std::error_code fresh_error;
        fresh = std::filesystem::directory_iterator(options.journal, fresh_error) ==
                    std::filesystem::directory_iterator() &&
                !fresh_error;
        std::ifstream known(config_directory() / "instances.jsonl");
        std::string line;
        while (std::getline(known, line)) {
            if (line.empty()) continue;
            replica::json::Value value;
            std::string name;
            std::string journal;
            std::string settings_path;
            std::string build;
            std::string store;
            std::string tools;
            unsigned long long card = 0u;
            if (!replica::json::parse(line, value) ||
                !replica::json::text(value, "name", name) ||
                !replica::json::text(value, "journal", journal)) {
                continue;
            }
            replica::json::text(value, "settings", settings_path);
            replica::json::text(value, "build", build);
            replica::json::text(value, "models", store);
            replica::json::text(value, "tools", tools);
            replica::json::number(value, "card", card);
            instances::Definition entry;
            entry.name = name;
            entry.journal = journal;
            entry.settings = settings_path;
            entry.build = build.empty() ? options.build : std::filesystem::path(build);
            entry.models = store;
            entry.tools = tools;
            entry.card = static_cast<unsigned>(card);
            for (const auto &member : value.members) {
                constexpr const char *prefix = "conversation_";
                if (member.first.rfind(prefix, 0u) != 0u ||
                    member.second.kind != replica::json::Kind::string) continue;
                const std::string number = member.first.substr(13u);
                unsigned agent = 0u;
                const auto parsed = std::from_chars(number.data(),
                                                     number.data() + number.size(), agent);
                if (!number.empty() && parsed.ec == std::errc() &&
                    parsed.ptr == number.data() + number.size()) {
                    entry.conversation_names[agent] = member.second.text;
                }
            }
            if (std::filesystem::path(journal) == options.journal) {
                for (const auto &name : entry.conversation_names) {
                    lifecycle.name_conversation(0u, name.first, name.second, false);
                }
                continue;
            }
            if (!lifecycle.seed(std::move(entry), true)) {
                std::fprintf(stderr, "AOTX-CTRL: %s\n", lifecycle.refusal().c_str());
            }
        }
    }
    shell::State shell_state;
    shell_state.show_wizard = !options.simulated && fresh;
    std::vector<chat::View> chat_views(simulated.conversations.size());
    instances::View instances_view;
    module::State module_view;
    sync::State sync_view;
    settings::State settings_view;
    control::LiveState live_control_view;
    browser::State browser_view;
    wizard::State wizard_view;
    wizard::LiveState live_wizard_view;
    wizard::DetectAction detect_action;
    monitor::Telemetry telemetry;
#ifdef AOTX_AFFECT
    affect::State affect_view;
    affect::dials::State dials_view;
#endif
    model::StoreAction model_action;
    model::LivePanelState model_panel;
    model::DetailsState model_details;
    voice::Queue speech;
    chat::persona::Store persona_store(config_directory() / "personas");
    toast::Lane toasts(&speech);
    if (!options.simulated) {
        replica::State *live = lifecycle.replica(0u);
        if (live == nullptr) return 2;
        std::snprintf(instances_view.build.data(), instances_view.build.size(), "%s",
                      options.build.c_str());
        std::snprintf(instances_view.models.data(), instances_view.models.size(), "%s",
                      live->models_directory().c_str());
        std::snprintf(live_wizard_view.build_path.data(), live_wizard_view.build_path.size(),
                      "%s", options.build.c_str());
        std::snprintf(live_wizard_view.models_path.data(), live_wizard_view.models_path.size(),
                      "%s", live->models_directory().c_str());
        std::snprintf(live_wizard_view.journal_path.data(),
                      live_wizard_view.journal_path.size(), "%s",
                      (options.journal.parent_path() / "first-journal").c_str());
        std::snprintf(live_wizard_view.settings_path.data(),
                      live_wizard_view.settings_path.size(), "%s",
                      (options.journal.parent_path() / "first-instance.settings").c_str());
        std::snprintf(live_wizard_view.instance_name.data(),
                      live_wizard_view.instance_name.size(), "%s", "First instance");
        const char *wizard_home = std::getenv("HOME");
        if (wizard_home != nullptr && wizard_home[0] != '\0') {
            std::snprintf(live_wizard_view.tools_path.data(),
                          live_wizard_view.tools_path.size(), "%s", wizard_home);
        }
    }
    if (!speech.enabled()) toasts.add(speech.refusal(), toast::Severity::warning, glfwGetTime());
    int frames = 0;
    std::size_t active_instance = static_cast<std::size_t>(-1);
    while (!glfwWindowShouldClose(window) &&
           (options.frame_limit < 0 || frames < options.frame_limit)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const double now = glfwGetTime();
        if (options.simulated) {
            speech.set_agent_count(simulated.agents.size());
            simulated.tick(now);
            for (std::string &result : simulated.take_results()) {
                toasts.add(std::move(result), toast::Severity::success, now);
            }
            shell::draw_dock_space(shell_state, simulated);
            if (shell_state.show_instances) {
                instances::draw(instances_view, simulated, toasts, now,
                                &shell_state.show_instances);
            }
            chat_views.resize(simulated.conversations.size());
            const std::size_t conversation_count = simulated.conversations.size();
            for (std::size_t index = 0; index < conversation_count; ++index) {
                if (simulated.conversations[index].window_open) {
                    chat::draw(chat_views[index], simulated, index, speech, now);
                }
            }
            if (shell_state.show_control) {
                control::draw(simulated, toasts, now, &shell_state.show_control);
            }
            if (shell_state.show_models) {
                model::draw(simulated, toasts, now, &shell_state.show_models);
            }
            if (shell_state.show_modules) {
                module::draw(module_view, simulated, toasts, now, &shell_state.show_modules);
            }
            if (shell_state.show_settings) {
                settings::draw(settings_view, simulated, toasts, now, &shell_state.show_settings);
            }
            if (shell_state.show_monitor) monitor::draw(simulated, &shell_state.show_monitor);
            if (shell_state.show_browser) {
                browser::draw(browser_view, simulated, &shell_state.show_browser);
            }
            if (shell_state.show_voice) voice::draw(speech, &shell_state.show_voice);
            wizard::draw(wizard_view, simulated, toasts, now, &shell_state.show_wizard);
        } else {
            lifecycle.tick(now);
            detect_action.tick();
            model_action.tick();
            for (std::string &result : lifecycle.take_results()) {
                const toast::Severity severity = result_severity(result);
                if (severity == toast::Severity::warning ||
                    severity == toast::Severity::error) {
                    toasts.add(std::move(result), severity, now);
                } else {
                    speech.speak(voice::Category::lifecycle,
                                 voice::Source::system(), result);
                    toasts.add(std::move(result), severity, now, 4.0, false);
                }
            }
            std::string model_result = model_action.take_result();
            if (!model_result.empty()) {
                const toast::Severity severity = result_severity(model_result);
                toasts.add(std::move(model_result), severity, now);
            }
            const std::size_t selected = lifecycle.selected();
            replica::State *live = lifecycle.replica(selected);
            client::Client *socket = lifecycle.client(selected);
            if (live == nullptr || socket == nullptr) continue;
            if (active_instance != selected) {
                active_instance = selected;
                chat_views.clear();
                live_control_view = control::LiveState{};
                model_panel = model::LivePanelState{};
                model_details.clear();
                module_view = module::State{};
                sync_view = sync::State{};
                settings_view = settings::State{};
                browser_view = browser::State{};
#ifdef AOTX_AFFECT
                affect_view = affect::State{};
                dials_view = affect::dials::State{};
#endif
            }
            std::size_t voice_agents = 1u;
            for (const replica::Agent &agent : live->agents()) {
                voice_agents = std::max(voice_agents,
                                        static_cast<std::size_t>(agent.id) + 1u);
            }
            speech.set_agent_count(voice_agents);
#ifdef AOTX_AFFECT
            speech.clear_affect_states();
            for (const replica::AffectTrace &row : live->affect_traces()) {
                speech.set_affect_state(row.agent, row.effective[0], row.effective[1]);
            }
#endif
            shell::draw_dock_space(shell_state, *live, *socket);
            if (shell_state.show_instances) {
                instances::draw(instances_view, lifecycle, toasts, now,
                                &shell_state.show_instances);
            }
            chat_views.resize(live->agents().size());
            for (std::size_t index = 0; index < live->agents().size(); ++index) {
                if (live->agents()[index].window_open) {
                    chat::draw(chat_views[index], *live, index, *socket, speech,
                               lifecycle, selected, persona_store,
                               shell_state.confidence_colors);
                }
            }
            if (shell_state.show_control) {
                control::draw(live_control_view, lifecycle, *live, *socket, toasts, now,
                              &shell_state.show_control);
            }
            const std::vector<instances::LiveInstance> model_instances = lifecycle.instances();
            const std::filesystem::path model_build = selected < model_instances.size()
                ? model_instances[selected].definition.build : options.build;
            model_details.update(model_build, live->models_directory(), live->models());
            if (shell_state.show_models) {
                model::draw(model_action, model_panel, model_details, model_build,
                            *live, *socket, toasts, now,
                            &shell_state.show_models);
            }
            model::draw_details(model_details);
            if (shell_state.show_modules) {
                module::draw(module_view, *live, *socket, toasts, now,
                             &shell_state.show_modules);
            }
            if (shell_state.show_sync) {
                sync::draw(sync_view, *live, *socket, &shell_state.show_sync);
            }
            if (shell_state.show_settings) {
                settings::draw(settings_view, *live, *socket, toasts, now,
                               &shell_state.show_settings);
            }
            if (shell_state.show_monitor) {
                monitor::draw(telemetry, *live, *socket, now, &shell_state.show_monitor);
            }
#ifdef AOTX_AFFECT
            if (shell_state.show_trace) {
                affect::draw(affect_view, *live, &shell_state.show_trace);
            }
            if (shell_state.show_dials) {
                affect::dials::draw(dials_view, *live, *socket, toasts, now,
                                    &shell_state.show_dials);
            }
#endif
            if (shell_state.show_browser) {
                browser::draw(browser_view, *live, &shell_state.show_browser);
            }
            if (shell_state.show_voice) voice::draw(speech, &shell_state.show_voice);
            wizard::draw(live_wizard_view, detect_action, model_action, lifecycle, *live,
                         toasts, now, &shell_state.show_wizard);
        }
        toasts.draw(now);

        ImGui::Render();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.035f, 0.043f, 0.052f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
            GLFWwindow *context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(context);
        }
        glfwSwapBuffers(window);
        ++frames;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    return 0;
}

} // namespace

int run(int argc, char **argv)
{
    Options options;
    std::error_code executable_error;
    options.build = std::filesystem::canonical(argv[0], executable_error).parent_path();
    if (executable_error) options.build = std::filesystem::path(argv[0]).parent_path();
    std::string layout_path;
    if (!chat::verify_key_paths()) {
        std::fputs("AOTX-CTRL refuses an invalid editor key path.\n", stderr);
        return 3;
    }
    if (!chat::format::verify_fixtures()) {
        std::fputs("AOTX-CTRL refuses an invalid reply format fixture.\n", stderr);
        return 3;
    }
    if (!chat::persona::verify_composition()) {
        std::fputs("AOTX-CTRL refuses an invalid persona composition.\n", stderr);
        return 3;
    }
    if (!sync::verify_state_logic()) {
        std::fputs("AOTX-CTRL refuses an invalid sync state fixture.\n", stderr);
        return 3;
    }
    if (!sim::verify_paths()) {
        std::fputs("AOTX-CTRL refuses an invalid simulated state path.\n", stderr);
        return 3;
    }
    if (!voice::verify_source_paths()) {
        std::fputs("AOTX-CTRL refuses an invalid voice assignment.\n", stderr);
        return 3;
    }
    if (!voice::verify_queue_rules()) {
        std::fputs("AOTX-CTRL refuses an invalid voice queue rule.\n", stderr);
        return 3;
    }
    if (!wizard::verify_gates()) {
        std::fputs("AOTX-CTRL refuses an invalid first-run gate.\n", stderr);
        return 3;
    }
    if (!monitor::verify_ring_figures()) {
        std::fputs("AOTX-CTRL refuses an invalid ring figure.\n", stderr);
        return 3;
    }
    if (!client::verify_frame()) {
        std::fputs("AOTX-CTRL refuses an invalid socket frame.\n", stderr);
        return 3;
    }
    if (!client::verify_outage()) {
        std::fputs("AOTX-CTRL refuses a repeated outage line.\n", stderr);
        return 3;
    }
    if (!replica::verify_fixtures()) {
        std::fputs("AOTX-CTRL refuses an invalid replica fixture.\n", stderr);
        return 3;
    }
    if (!parse_options(argc, argv, options)) {
        return 2;
    }
    if (options.help) {
        print_usage();
        return 0;
    }
    if (!make_layout_path(layout_path)) {
        return 2;
    }

    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fputs("AOTX-CTRL cannot start GLFW.\n", stderr);
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow *window = glfwCreateWindow(1280, 800, "AOTX-CTRL", nullptr, nullptr);
    if (window == nullptr) {
        std::fputs("AOTX-CTRL cannot open its main window.\n", stderr);
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    const int result = run_loop(window, options, layout_path);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}

} // namespace aotx::ctrl::app

int main(int argc, char **argv)
{
    return aotx::ctrl::app::run(argc, argv);
}
