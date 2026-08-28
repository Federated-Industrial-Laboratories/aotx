/* Purpose: Hold the cell grid of the text display and the rectangle of each panel.
 * Owns: The cell grid and the panel table.
 * Launch shape: Not applicable; state that the panel kernels write.
 * Lifetime: The whole run. */
#include "ui/ui.cuh"

__device__ aotx_ui_cell aotx_ui_grid[AOTX_UI_CELLS];

/* The layout is fixed for this version. The left column holds the console above the bus
 * messages. The right column holds the agents, the memory, the tick and the seam. */
__constant__ aotx_ui_panel aotx_ui_panel_table[AOTX_UI_PANELS] = {
    {   0u,  0u, 100u, 34u },   /* console, with the command line on its last row */
    { 100u,  0u,  60u, 16u },   /* agents */
    {   0u, 34u, 100u, 16u },   /* bus */
    { 100u, 16u,  60u, 12u },   /* arena */
    { 100u, 28u,  60u, 12u },   /* tick, with the agent counts on its last row */
    { 100u, 40u,  60u, 10u },   /* seam */
};

/* The layout covers the grid exactly, so no cell belongs to two panels or to none. */
typedef char aotx_ui_check_left[(34u + 16u == AOTX_UI_ROWS) ? 1 : -1];
typedef char aotx_ui_check_right[(16u + 12u + 12u + 10u == AOTX_UI_ROWS) ? 1 : -1];
typedef char aotx_ui_check_cols[(100u + 60u == AOTX_UI_COLS) ? 1 : -1];

/* The agents panel holds a title row, a row of column names, the agent rows, the name row
 * of the requests and the request rows. The check refuses a count that does not fit. */
typedef char aotx_ui_check_agents[(3u + AOTX_UI_AGENT_ROWS + AOTX_UI_REQUEST_ROWS == 16u)
                                  ? 1 : -1];
