// Purpose: Start and stop the control program.
// Owns: Program arguments and the main user interface loop.
// Launch shape: One process runs one user interface thread.
// Lifetime: The application owns all resources until exit.
#ifndef AOTX_CTRL_APP_HPP
#define AOTX_CTRL_APP_HPP

namespace aotx::ctrl::app {

int run(int argc, char **argv);

} // namespace aotx::ctrl::app

#endif
