/* Purpose: Prove the parity gate refuses a help line that the parser does not dispatch.
 * Owns: Nothing; a fixture of the parity gate, read by no program of the system.
 * Threading: Not applicable; constant text.
 * Lifetime: The check that runs the gate over it. */
#ifndef AOTX_CLI_HELP_FIXTURE_H
#define AOTX_CLI_HELP_FIXTURE_H

/* The last line names a command that no branch of the parser takes. */
static const char *aotx_test_help_line(unsigned int index)
{
    switch (index) {
    case 0u: return "commands:";
    case 1u: return "  help                     show these lines";
    case 2u: return "  quit                     stop the run";
    default: return "  restart                  start the run again";
    }
}

#endif
