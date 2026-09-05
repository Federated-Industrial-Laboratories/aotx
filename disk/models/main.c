/* Purpose: List, inspect, fetch, check, activate, and remove model files.
 * Owns: One catalog table and one store view for the command.
 * Threading: One process and one command.
 * Lifetime: The command. */
#include "disk/models/models.h"
#include "disk/models/inspect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef AOTX_MODELS_CATALOG
#define AOTX_MODELS_CATALOG "share/models/catalog.jsonl"
#endif

#ifndef AOTX_MODELS_DIRECTORY
#define AOTX_MODELS_DIRECTORY "models"
#endif

static void usage(void)
{
    fprintf(stderr, "usage: aotx_models [--dir <dir>] [--catalog <file>] list\n");
    fprintf(stderr, "       aotx_models [--dir <dir>] [--catalog <file>] fetch <name>\n");
    fprintf(stderr, "       aotx_models [--dir <dir>] check\n");
    fprintf(stderr, "       aotx_models [--dir <dir>] [--catalog <file>] activate <role> <name>\n");
    fprintf(stderr, "       aotx_models [--dir <dir>] [--catalog <file>] remove <name>\n");
    fprintf(stderr, "       aotx_models inspect <file-or-url>\n");
}

static int list_store(const char *dir, const aotx_model_catalog *catalog)
{
    aotx_model_view views[AOTX_MODEL_CATALOG_MAX * 2u];
    char reason[256];
    int count = aotx_model_store_scan(dir, catalog, views,
                                      AOTX_MODEL_CATALOG_MAX * 2u,
                                      reason, sizeof(reason));
    int i;
    if (count < 0) {
        fprintf(stderr, "aotx_models: %s\n", reason);
        return 1;
    }
    printf("state\tname\trole\tquant\tbytes\tsource\tverified\n");
    for (i = 0; i < count; i++) {
        const aotx_model_view *view = &views[i];
        uint64_t bytes = (view->bytes_on_disk != 0u)
                       ? view->bytes_on_disk : view->catalog.bytes;
        printf("%s\t%s\t%s\t%s\t%llu\t%s\t%s\n",
               aotx_model_state_text(view->state), view->catalog.name,
               view->catalog.role, view->catalog.quant,
               (unsigned long long)bytes, view->catalog.source,
               view->verified ? "true" : "false");
    }
    return 0;
}

int main(int argc, char **argv)
{
    aotx_model_catalog catalog;
    const aotx_model_catalog_entry *entry;
    const char *dir = AOTX_MODELS_DIRECTORY;
    const char *catalog_path = AOTX_MODELS_CATALOG;
    const char *command = NULL;
    const char *arg[2] = { NULL, NULL };
    char reason[512];
    int arg_count = 0;
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            dir = argv[++i];
        } else if (strcmp(argv[i], "--catalog") == 0 && i + 1 < argc) {
            catalog_path = argv[++i];
        } else if (command == NULL) {
            command = argv[i];
        } else if (arg_count < 2) {
            arg[arg_count++] = argv[i];
        } else {
            usage();
            return 2;
        }
    }
    if (command == NULL) {
        usage();
        return 2;
    }
    if (strcmp(command, "inspect") == 0) {
        if (arg_count != 1) { usage(); return 2; }
        return aotx_model_inspect(arg[0]);
    }
    if (strcmp(command, "check") == 0) {
        int checked;
        if (arg_count != 0) {
            usage();
            return 2;
        }
        checked = aotx_model_store_check(dir, reason, sizeof(reason));
        if (checked < 0) {
            fprintf(stderr, "aotx_models: %s\n", reason);
            return 1;
        }
        printf("checked %d model files\n", checked);
        return 0;
    }
    if (aotx_model_catalog_read(catalog_path, &catalog, reason, sizeof(reason)) < 0) {
        fprintf(stderr, "aotx_models: %s\n", reason);
        return 1;
    }
    if (strcmp(command, "list") == 0) {
        if (arg_count != 0) {
            usage();
            return 2;
        }
        return list_store(dir, &catalog);
    }
    if (strcmp(command, "fetch") == 0) {
        if (arg_count != 1) {
            usage();
            return 2;
        }
        entry = aotx_model_catalog_find(&catalog, arg[0]);
        if (entry == NULL || aotx_model_fetch(dir, entry, AOTX_MODEL_FETCH_TIMEOUT,
                                              reason, sizeof(reason)) != 0) {
            fprintf(stderr, "aotx_models: %s\n",
                    (entry == NULL) ? "the catalog has no such model" : reason);
            return 1;
        }
        printf("fetched %s\n", entry->name);
        return 0;
    }
    if (strcmp(command, "activate") == 0) {
        if (arg_count != 2) {
            usage();
            return 2;
        }
        entry = aotx_model_catalog_find(&catalog, arg[1]);
        if (entry == NULL || aotx_model_store_activate(dir, entry, arg[0],
                                                        reason, sizeof(reason)) != 0) {
            fprintf(stderr, "aotx_models: %s\n",
                    (entry == NULL) ? "the catalog has no such model" : reason);
            return 1;
        }
        printf("activated %s as %s\n", entry->name, arg[0]);
        return 0;
    }
    if (strcmp(command, "remove") == 0) {
        if (arg_count != 1) {
            usage();
            return 2;
        }
        entry = aotx_model_catalog_find(&catalog, arg[0]);
        if (entry == NULL || aotx_model_store_remove(dir, entry, reason, sizeof(reason)) != 0) {
            fprintf(stderr, "aotx_models: %s\n",
                    (entry == NULL) ? "the catalog has no such model" : reason);
            return 1;
        }
        printf("removed %s\n", entry->name);
        return 0;
    }
    usage();
    return 2;
}
