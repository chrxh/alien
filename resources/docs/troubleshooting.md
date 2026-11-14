# Troubleshooting

Solutions for the most common problems. If your problem is not listed, ask on the [Discord server](https://discord.gg/7bjyZdXXQ2) or open an issue on [GitHub](https://github.com/chrxh/alien/issues).

## ALIEN does not start or crashes

Please make sure that:

1. You have a supported graphics card. For NVIDIA, this is a GPU with compute capability 7.5 or higher, for example from the GeForce GTX 16 series, the RTX 20 series or newer. For AMD, it is a GPU of the RDNA2 generation or newer, for example from the Radeon RX 6000 series. Support for AMD GPUs is experimental, so ALIEN may run slowly or crash on them. On Windows, *alien.exe* is for NVIDIA and *alien-amd.exe* for AMD.
2. The latest graphics driver is installed.
3. The installation folder and its parent folders contain no non-English characters. On Windows, the user name should not contain such characters either.
4. ALIEN has write access to its own folder.
5. ALIEN is started from its own folder, for example with the Start menu entry or with a double click on the program file. Otherwise it does not find its *resources* folder and closes right away.
6. With several graphics cards, the primary monitor is connected to the card that ALIEN uses. ALIEN computes on a dedicated card and prefers the one with the highest compute capability. It also renders on this card if the card can show the window, which is the case when the primary monitor is connected to it. Otherwise ALIEN renders on another card, which slows it down.
7. On computers with integrated and dedicated graphics, ALIEN runs on the dedicated card. On Windows, you can set this in the *Graphics settings* by adding *alien.exe*, or *alien-amd.exe* for an AMD card, and choosing *High performance*.

If the problem persists, enable **Settings > Debug mode**, reproduce the problem and attach *log.txt* and *trace.txt* from the ALIEN folder to a GitHub issue. ALIEN replaces *log.txt* at every start, so copy the file before you start ALIEN again.

ALIEN switches **Debug mode** off again at the next start. If ALIEN crashes before you can use the menu, start it with the option -d to switch on debug mode from the beginning. To do this, open a command prompt in the ALIEN folder and enter `alien.exe -d`, or `alien-amd.exe -d` for AMD.

## The simulation is slow

- Large worlds with many objects need a powerful graphics card. Try a smaller simulation.
- Rendering also uses the graphics card. Lower the **Frames per second** in **Settings > Display settings**, hide the simulation with **Alt+I**, or switch to console mode with **Alt+C**.
- Check that **Slow down** and **Sync with rendering** in the **Temporal control** window are switched off.
- An exploding population fills the world with cells and slows the simulation down. See [Energy](energy.md#troubleshooting-energy).

## The user interface is too small or too large

Open **Settings > Display settings**. **Adopt scaling from OS** is enabled by default and takes the scaling from the operating system. To set the scaling yourself, disable it and change **Content scaling**, which ranges from 1 to 4. Then click **Adopt** and confirm with **Yes**. ALIEN saves the setting and closes, and the new scaling applies when you start ALIEN again.

## My creatures do not move

- Muscles need a signal. Switch to edit mode with **Alt+E**, select the muscle cells and press **Alt+N**. In the inspection window, open **Neural activity** and check that **Signal #0** is not 0.
- Signals flow from the last node of a gene towards node 0. A sensor or other signal source belongs at the end of the gene.
- Muscles in *Auto bending* mode need a signal in channel #0 that stays either positive or negative. A negative value reverses the thrust, so a square signal that alternates between positive and negative values hardly moves a creature. Channel #1 gives the desired direction, and muscles that would push away from it reduce or stop their work. See [Muscle](cell-types.md#muscle).
- Sensors, muscles and communicators need the front direction of their creature. The head cell sends it out every 100 time steps, so newly built cells can stay idle for up to about 100 time steps.
- Cells under construction are inactive. They become active when the last node of their gene is built. With several concatenations or branches, each copy of the gene becomes active as soon as its own last node is built. Body parts that a creature builds later with its own constructors stay inactive until they are complete, while the rest of the creature already acts.

## My seeds do not build anything

- A seed created with **Create seed** has only the normal energy of a cell and needs more energy before it can build. Use **Create seed with energy**, or fill the external energy pool and make sure that **Inflow for constructors** is greater than zero.
- **Inflow threshold factor** should be 0, which is the default. With a higher value, a seed requests energy from the pool only after it has gathered this fraction of the required energy by itself, for example from energy particles.
- Constructors without an **Auto trigger interval** wait for a signal in channel #0.
- A construction is postponed if the new connection would cross other connections nearby, for example when the seed sits inside dense matter. Move the seed to a free spot.

## The population explodes or dies out

See [Energy](energy.md#troubleshooting-energy) and [Evolution experiments](evolution.md#ingredient-3-selection).

## The browser stays empty

- Check your internet connection.
- Check the server address in **Settings > Network settings**.
- Click **Refresh** in the tool bar of the browser.

## My AI agent cannot connect

- In the **MCP server** window (Alt+6), check that the status is *Running*.
- The agent must connect to ALIEN from the same computer. Chat applications that connect through their provider's servers on the internet cannot reach ALIEN.
- The URL configured in the agent must match the one shown in the **MCP server** window, including the port.
- If another program uses the port, choose a different port with the **Settings** button of the **MCP server** window.
- Some agents need to be restarted after adding a new MCP server.

## I lost my simulation

- If **Settings > Save on exit** is enabled, which is the default, ALIEN saves the simulation when you close the program normally and opens it again at the next start. After a crash there is no such save, so ALIEN opens the simulation of the last normal exit instead.
- **Autosaves** in the left column of the browser shows the autosaves. Autosaves exist only if you created them with **Autosave now** or switched on automatic saving with **Enable autosave** in the right column. Automatic saving is off by default. Save points of earlier versions appear there as well if their folder lies in the folder of the simulations of this computer.
- The flashback of the **Temporal control** window exists only in memory and is lost when ALIEN is closed.

## The documentation is missing

ALIEN loads the documentation from *resources/docs* in the folder from which it was started, not from the folder of the program file. Check that the ALIEN folder contains *resources/docs* with the file *contents.md*, and start ALIEN from this folder, for example with the Start menu entry.
