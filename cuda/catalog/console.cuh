/* Purpose: Give the console the commands that read the catalog and take a module out.
 * Owns: Nothing; the catalog and the console buffer hold the state.
 * Launch shape: One thread; the apply step calls these in slot order.
 * Lifetime: The whole run. */
#ifndef AOTX_CATALOG_CONSOLE_CUH
#define AOTX_CATALOG_CONSOLE_CUH

#include "catalog/catalog.cuh"
#include "cli/cli.cuh"

/* Write one line for each module of the catalog: the name, the kind, the state and the
 * version. A refused entry carries its reason. A kind of zero shows every kind. */
__device__ void aotx_catalog_modules_command(aotx_cli_out *out, unsigned int kind);

/* Write one module in full: the manifest, and the first lines of the body of a skill. */
__device__ void aotx_catalog_module_command(aotx_cli_out *out, const char *name,
                                            unsigned int length);

/* Act on a remove line. The command builds a remove record and publishes it as a class A
 * record of the console. A name the catalog refuses writes the reason and changes
 * nothing.
 *
 * A replay of the journal makes the change from the record and not from the line. A
 * restored run therefore folds each remove body one time. */
__device__ void aotx_catalog_remove_command(aotx_cli_out *out, const char *name,
                                            unsigned int length, unsigned long long tick);

/* Act on an import line of a surface the feeder does not read, such as the window. The
 * command writes one request record, which the drain gives the feeder, and the feeder
 * reads the directory and publishes the import. The record is derived, so a replay of the
 * journal writes none and the feeder reads no directory a second time. */
__device__ void aotx_catalog_import_command(aotx_cli_out *out, const char *path,
                                            unsigned int length, unsigned long long tick);

/* Show the report of an import the feeder refused. The feeder publishes that report as an
 * input line. The line then takes the path of every operator line: one console line and one
 * bus note. A replay writes both again and changes nothing else. */
__device__ void aotx_catalog_import_said(aotx_cli_out *out, const char *text,
                                         unsigned int length, unsigned long long tick);

/* Write one line for each name of a tools list or a skills list that the catalog does not
 * hold. The commit of an import calls this, so the operator reads which names it lost. */
__device__ void aotx_catalog_unknown_names(const aotx_catalog_entry *row,
                                           aotx_cli_out *out);

#endif
