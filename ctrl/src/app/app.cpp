// Purpose: Run GLFW, OpenGL, ImGui, and all control shell modules.
// Owns: Window resources, ImGui resources, and the frame loop.
// Launch shape: One process draws one frame at a time.
// Lifetime: Resources start in order and stop in reverse order.
#include "app/app.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "browser/browser.hpp"
#include "chat/chat.hpp"
#include "client/client.hpp"
#include "control/control.hpp"
#include "imgui.h"
#include "instances/instances.hpp"
#include "model/model.hpp"
#include "module/module.hpp"
#include "monitor/monitor.hpp"
#include "replica/replica.hpp"
#include "settings/settings.hpp"
#include "shell/shell.hpp"
#include "sim/sim.hpp"
#include "theme/theme.hpp"
#include "toast/toast.hpp"
#include "voice/voice.hpp"
#include "wizard/wizard.hpp"

#include <GLFW/glfw3.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace aotx::ctrl::app {
namespace {

struct Options {
    int frame_limit = -1;
    bool simulated = false;
    std::filesystem::path journal;
    std::filesystem::path settings;
    std::filesystem::path build;
};

void glfw_error(int, const char *description)
{
    std::fprintf(stderr, "GLFW reports: %s\n", description);
}

bool parse_options(int argc, char **argv, Options &options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
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
        if (options.settings.empty() ||
            !replica::setting_value(options.settings, "journal.dir", journal) || journal.empty()) {
            std::fputs("AOTX-CTRL needs a journal directory in live mode.\n", stderr);
            return false;
        }
        options.journal = journal;
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
    std::unique_ptr<replica::State> live;
    std::unique_ptr<client::Client> socket;
    instances::Lifecycle lifecycle;
    if (!options.simulated) {
        live = std::make_unique<replica::State>(options.journal, options.settings);
        if (!live->open()) {
            for (const std::string &result : live->take_results()) {
                std::fprintf(stderr, "AOTX-CTRL: %s\n", result.c_str());
            }
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            return 2;
        }
        socket = std::make_unique<client::Client>(options.journal);
        instances::Definition local;
        local.name = "Local instance";
        local.journal = options.journal;
        local.settings = live->settings();
        local.build = options.build;
        local.models = live->models_directory();
        lifecycle.seed(std::move(local));
    }
    shell::State shell_state;
    std::vector<chat::View> chat_views(simulated.conversations.size());
    instances::View instances_view;
    module::State module_view;
    settings::State settings_view;
    control::LiveState live_control_view;
    browser::State browser_view;
    wizard::State wizard_view;
    wizard::LiveState live_wizard_view;
    wizard::DetectAction detect_action;
    monitor::Telemetry telemetry;
    model::StoreAction model_action;
    voice::Queue speech;
    toast::Lane toasts(&speech);
    if (!options.simulated) {
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
    }
    if (!speech.enabled()) toasts.add(speech.refusal(), toast::Severity::warning, glfwGetTime());
    int frames = 0;
    while (!glfwWindowShouldClose(window) &&
           (options.frame_limit < 0 || frames < options.frame_limit)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const double now = glfwGetTime();
        if (options.simulated) {
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
            wizard::draw(wizard_view, simulated, toasts, now, &shell_state.show_wizard);
        } else {
            live->tick(now);
            socket->tick(now);
            lifecycle.tick(now);
            detect_action.tick();
            model_action.tick();
            for (std::string &result : live->take_results()) {
                const toast::Severity severity = result_severity(result);
                toasts.add(std::move(result), severity, now);
            }
            for (std::string &result : socket->take_results()) {
                toasts.add(std::move(result), toast::Severity::warning, now);
            }
            for (std::string &result : lifecycle.take_results()) {
                const toast::Severity severity = result_severity(result);
                toasts.add(std::move(result), severity, now);
            }
            std::string model_result = model_action.take_result();
            if (!model_result.empty()) {
                const toast::Severity severity = result_severity(model_result);
                toasts.add(std::move(model_result), severity, now);
            }
            shell::draw_dock_space(shell_state, *live, *socket);
            if (shell_state.show_instances) {
                instances::draw(instances_view, lifecycle, toasts, now,
                                &shell_state.show_instances);
            }
            chat_views.resize(live->agents().size());
            for (std::size_t index = 0; index < live->agents().size(); ++index) {
                if (live->agents()[index].window_open) {
                    chat::draw(chat_views[index], *live, index, *socket, speech);
                }
            }
            if (shell_state.show_control) {
                control::draw(live_control_view, *live, *socket, toasts, now,
                              &shell_state.show_control);
            }
            if (shell_state.show_models) {
                model::draw(model_action, options.build, *live, *socket, toasts, now,
                            &shell_state.show_models);
            }
            if (shell_state.show_modules) {
                module::draw(module_view, *live, *socket, toasts, now,
                             &shell_state.show_modules);
            }
            if (shell_state.show_settings) {
                settings::draw(settings_view, *live, *socket, toasts, now,
                               &shell_state.show_settings);
            }
            if (shell_state.show_monitor) {
                monitor::draw(telemetry, *live, *socket, now, &shell_state.show_monitor);
            }
            if (shell_state.show_browser) {
                browser::draw(browser_view, *live, &shell_state.show_browser);
            }
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
    if (!sim::verify_paths()) {
        std::fputs("AOTX-CTRL refuses an invalid simulated state path.\n", stderr);
        return 3;
    }
    if (!voice::verify_source_paths()) {
        std::fputs("AOTX-CTRL refuses an invalid voice assignment.\n", stderr);
        return 3;
    }
    if (!client::verify_frame()) {
        std::fputs("AOTX-CTRL refuses an invalid socket frame.\n", stderr);
        return 3;
    }
    if (!replica::verify_fixtures()) {
        std::fputs("AOTX-CTRL refuses an invalid replica fixture.\n", stderr);
        return 3;
    }
    if (!parse_options(argc, argv, options) || !make_layout_path(layout_path)) {
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
