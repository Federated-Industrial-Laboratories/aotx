/* Purpose: Give the lines of the help command.
 * Owns: The help text; nothing else.
 * Launch shape: A device function; the parser calls it once for each line.
 * Lifetime: The whole run; the text is constant. */
#ifndef AOTX_CLI_HELP_CUH
#define AOTX_CLI_HELP_CUH

#include "cli/cli.cuh"

/* One line of the help text. The lines are in the register the documentation uses. */
static __device__ __forceinline__ const char *aotx_cli_help_line(unsigned int index)
{
    switch (index) {
    case 0u: return "commands:";
    case 1u: return "  help                     show these lines";
    case 2u: return "  bus [kind]               show the last bus messages of a kind";
    case 3u: return "  note <text>              put a note on the bus";
    case 4u: return "  finding <source> <text>  put a finding on the bus";
    case 5u: return "      a source is computed, fetched, recalled or testimony";
    case 6u: return "  say <text>               send a message to the console agent";
    case 7u: return "  stop                     end the reply that runs";
    case 8u: return "  spawn <role> [n]         make n agents of a role; n is 1 to 8";
    case 9u: return "      a role is the name of a role module of the catalog";
    case 10u: return "  task <agent|role> <text> [verify]   open a task for an agent";
    case 11u: return "      verify as the last word asks a verifier to judge the result";
    case 12u: return "  authorise <id>           let a tool request of that number run";
    case 13u: return "  refuse <id>              stop a tool request of that number";
    case 14u: return "  mem                      show the memory regions and the budget";
    case 15u: return "  agents                   show the agents";
    case 16u: return "  stats                    show the counts of the last tick";
    case 17u: return "  settings                 show the settings and when each takes "
                     "effect";
    case 18u: return "  set <key> <value>        change one setting";
    case 19u: return "  modules [kind]           show the catalog; a kind is skill, role "
                     "or tool";
    case 20u: return "  module <name>            show one module in full";
    case 21u: return "  skills                   show the skills of the catalog";
    case 22u: return "  roles                    show the roles of the catalog";
    case 23u: return "  tools                    show the tools of the catalog";
    case 24u: return "  import <path>            put the module of a directory in the "
                     "catalog";
    case 25u: return "  remove <name>            take one module out of the catalog";
    default: return "  quit                     stop the run";
    }
}


#endif
