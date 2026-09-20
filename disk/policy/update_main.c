/* Purpose: Create one explicitly selected compatible runtime policy revision.
 * Owns: Command arguments; the converter owns all file leases.
 * Threading: One disk process; no code is executed from policy bundles.
 * Lifetime: A separate complete output is published after validation. */
#include "disk/runtime/policy.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *source = NULL, *expected = NULL, *policy = NULL, *output = NULL, *mapping = NULL;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) goto usage;
        const char **target = NULL;
        if (!strcmp(argv[i], "--runtime")) target = &source;
        else if (!strcmp(argv[i], "--from")) target = &expected;
        else if (!strcmp(argv[i], "--policy")) target = &policy;
        else if (!strcmp(argv[i], "--output")) target = &output;
        else if (!strcmp(argv[i], "--state-map")) target = &mapping;
        if (!target || *target) goto usage;
        *target = argv[i + 1];
    }
    if (!source || !expected || !policy || !output || !mapping || strcmp(mapping, "preserve")) goto usage;
    int rc = aotx_runtime_policy_update(source, expected, policy, output);
    if (rc) fprintf(stderr, "policy update refused: %s (%d)\n", aotx_ccir_status_text(rc), rc);
    else puts("policy update saved; activation requires the selected native policy trust");
    return rc ? 1 : 0;
usage:
    fputs("Use: aotx_policy_update --runtime FILE --from SHA256 --policy FILE --state-map preserve --output FILE\n"
          "Create a separate complete file. The input file stays unchanged.\n"
          "Preserve requires the same policy ABI, state schema, and state size.\n"
          "Use preserve only when the new policy assigns the same meaning to each state byte.\n"
          "An unsupported conversion or an existing output is refused.\n", stderr);
    return 2;
}
