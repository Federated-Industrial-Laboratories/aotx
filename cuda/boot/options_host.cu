/* Purpose: Read the command line of the boot program and write its options.
 * Owns: Nothing; the caller holds the options.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: The start of the program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/boot.cuh"

void aotx_boot_usage(void)
{
    printf("aotx_boot --journal <dir> [--models <dir>] [--roles <list>] [--restore]\n");
    printf("          [--window] [--ticks <n>] [--workload <n>] [--blocks <n>]\n");
    printf("          [--records <n>] [--derive <list>] [--root <dir>] [--solo]\n");
    printf("          [--clock-only]\n");
    printf("  --journal    the directory the journal goes in\n");
    printf("  --models     the directory the model files are in\n");
    printf("  --roles      roles of the model file list to load, with commas between\n");
    printf("  --root       the one directory a file read tool may reach\n");
    printf("  --restore    replay the journal before the first input\n");
    printf("  --window     show the panels in a window on the display\n");
    printf("  --ticks      run this many ticks, then stop; zero runs on\n");
    printf("  --workload   records the tick load writes for each tick\n");
    printf("  --blocks     blocks of the tick load\n");
    printf("  --records    stop a run that has no tick count at this record count\n");
    printf("  --derive     types the drain makes lines from, with commas between them\n");
    printf("  --solo       run with no disk side programs\n");
    printf("  --clock-only run the clock module check and stop\n");
}

int aotx_boot_parse(int argc, char **argv, aotx_boot_options *options)
{
    memset(options, 0, sizeof *options);
    options->records = 1000000ull;
    options->blocks = 64u;
    for (int i = 1; i < argc; ++i) {
        int last = (i + 1 >= argc);
        if (strcmp(argv[i], "--journal") == 0 && !last) {
            options->journal = argv[++i];
        } else if (strcmp(argv[i], "--models") == 0 && !last) {
            options->models = argv[++i];
        } else if (strcmp(argv[i], "--roles") == 0 && !last) {
            options->roles = argv[++i];
        } else if (strcmp(argv[i], "--root") == 0 && !last) {
            options->root = argv[++i];
        } else if (strcmp(argv[i], "--ticks") == 0 && !last) {
            options->ticks = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--workload") == 0 && !last) {
            options->workload = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--records") == 0 && !last) {
            options->records = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--derive") == 0 && !last) {
            options->derive = argv[++i];
        } else if (strcmp(argv[i], "--blocks") == 0 && !last) {
            options->blocks = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--restore") == 0) {
            options->restore = 1;
        } else if (strcmp(argv[i], "--window") == 0) {
            options->window = 1;
        } else if (strcmp(argv[i], "--solo") == 0) {
            options->solo = 1;
        } else if (strcmp(argv[i], "--clock-only") == 0) {
            options->clock_only = 1;
        } else {
            fprintf(stderr, "the option %s is not known\n", argv[i]);
            aotx_boot_usage();
            return 2;
        }
    }
    return 0;
}

