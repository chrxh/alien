# Troubleshooting

Solutions for the most common problems. If your problem is not listed, ask on the [Discord server](https://discord.gg/7bjyZdXXQ2) or open an issue on [GitHub](https://github.com/chrxh/alien/issues).

## ALIEN does not start or crashes

Please make sure that:

1. You have a supported graphics card: an NVIDIA GPU with compute capability 7.5 or higher, for example a GeForce RTX 20 series or newer, or an AMD GPU of the RDNA2 generation or newer. On Windows, *alien.exe* is for NVIDIA and *alien-amd.exe* for AMD.
2. The latest graphics driver is installed.
3. The installation folder and its parent folders contain no non-English characters. On Windows, the user name should not contain such characters either.
4. ALIEN has write access to its own folder.
5. With several graphics cards, the primary monitor is connected to the card that ALIEN uses. ALIEN computes and renders on the same card.
6. On computers with integrated and dedicated graphics, ALIEN runs on the dedicated card. On Windows, open the *Graphics settings*, add *alien.exe* and choose *High performance*.

If the problem remains, enable **Settings > Debug mode**, reproduce the error and attach *log.txt* and the trace file from the ALIEN folder to a GitHub issue.

## The simulation is slow

- Large worlds with many objects need a powerful graphics card. Try a smaller simulation.
- Rendering also uses the graphics card. Lower the **Frames per second** in **Settings > Display settings**, hide the simulation with **Alt+I**, or switch to the console mode with **Alt+C**.
- Check that **Slow down** and **Sync with rendering** in the window **Temporal control** are switched off.
- An exploding population fills the world with cells. See [Energy](energy.md#troubleshooting-energy).

## The user interface is too small or too large

Open **Settings > Display settings** and change the **Content scaling**, or enable **Adopt scaling from OS**.

## My creatures do not move

- Muscles need a signal. Check in the inspection window (Alt+N) whether channel #0 of the muscle cells is different from 0.
- Signals flow from the last node of a gene towards node 0. A sensor or other signal source belongs at the end of the gene.
- Muscles in the auto bending mode need a positive signal in channel #0. A square signal that alternates its sign hardly moves a creature. See [Muscle](cell-types.md#muscle).
- Sensors, muscles and communicators only work in cells that belong to a creature with a head cell, because they need the front direction.
- Cells under construction are inactive. A creature starts to act only after its last cell is built.

## My seeds do not build anything

- A seed created with **Create seed** needs energy. Use **Create seed with energy**, or fill the external energy pool and make sure that **Inflow for constructors** is greater than zero.
- Constructors with an empty **Auto trigger interval** wait for a signal in channel #0.
- A construction is postponed if the new connection would cross other connections nearby, for example when the seed sits inside dense matter. Move the seed to a free spot.

## The population explodes or dies out

See [Energy](energy.md#troubleshooting-energy) and [Evolution experiments](evolution.md#ingredient-3-selection).

## The browser stays empty

- Check your internet connection.
- Check the server address in **Settings > Network settings**.
- Click **Refresh** in the tool bar of the browser.

## My AI agent cannot connect

- Check in the window **MCP server** (Alt+6) that the status is *Running*.
- The agent must run on the same computer, and the URL in the agent must match the one shown in the window, including the port.
- If another program uses the port, choose a different port with the **Settings** button of the MCP server window.
- Some agents need to be restarted after adding a new MCP server.

## I lost my simulation

- If **Settings > Save on exit** is enabled, the last simulation is restored at the next start.
- The window **Autosave** (Alt+5) lists the save points that were created while the simulation was running.
- A flashback in the window **Temporal control** only lives in memory and is lost when ALIEN closes.

## The documentation is missing

The documentation is loaded from the folder *resources/docs* next to the program. Start ALIEN from its own folder so that it finds the *resources* folder.
