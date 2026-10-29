# Files and command line

This chapter describes the files ALIEN works with, how simulations are saved automatically, how to share your work through the browser and how to run simulations without a window.

## File types

| File | Contents |
| --- | --- |
| *.sim* | A complete simulation: all objects, energy particles, creatures and genomes, the simulation parameters, the statistics history, the time step and the position of the view. |
| *.genome* | A single genome, as created in the genome editor. |
| *.settings.json* | A set of simulation parameters including layers and radiation sources. It can be loaded into any simulation. |
| *.png* | A picture created with **Simulation > Save picture**. |
| *.mp4* | A video recorded with **Tools > Recorder**. |
| *.flight.json* | The waypoints and settings of a camera flight, as created in **Tools > Recorder**. |

## Opening and saving

- **Simulation > Open** (Ctrl+O) loads a simulation file and **Simulation > Save** (Ctrl+S) saves the current simulation.
- **Simulation > New** (Ctrl+N) creates an empty world.
- Genomes are opened and saved in the tool bar of the genome editor.
- Parameter sets are opened and saved in the tool bar of the window **Simulation parameters**.
- Camera flights are opened and saved in the tool bar of the window **Recorder**.
- If **Settings > Save on exit** is enabled, ALIEN saves the current simulation when it closes and opens it again at the next start.

## Autosave and save points

The window **Autosave** (Alt+5) protects long runs. It creates save points of the running simulation at regular intervals:

- **Autosave interval** sets the time between two save points in minutes.
- **Number of files** limits how many save points are kept. The oldest ones are deleted.
- **Directory** sets where the save points are stored.
- The buttons create a save point by hand or delete save points. The download button in a row loads that save point.

## Sharing in the browser

The browser (Alt+B) connects ALIEN with the ALIEN server, where users share simulations and genomes.

- **Browsing** works without an account. The workspace **Featured** contains simulations of the ALIEN project, **Community** contains everything users have shared.
- **Logging in.** Choose **Network > Login** (Alt+L) or click the account chip in the browser. There you can also create a new account. With an account you get a **Private** workspace, can upload your own work and can react to the work of others.
- **Uploading.** **Network > Upload simulation** (Alt+D) uploads the current simulation, **Network > Upload genome** (Alt+Q) uploads the genome of the genome editor. The name is prefilled with your user name as folder followed by the project name of the simulation or the name of the genome, for example *UserName/Ocean*. You can add a description. **Share with the community** makes the upload visible to everyone, **Keep private** shows it only in your **Private** workspace. A slash in the name creates folders, for example *My worlds/Ocean*. After the upload, **Show in browser** selects the new entry in the browser.
- **Managing.** The browser tool bar lets you rename your uploads, replace them by a newer version, switch them between private and community and delete them.
- **Reacting.** Click the reaction field of a simulation to give it a reaction. The gallery view can sort by reactions, date or downloads.

## Command line interface

ALIEN includes a second program, *alien-cli* (*alien-cli.exe* on Windows), which runs a simulation without any window. It is useful for long experiments, for automation and on computers without a monitor, such as rented cloud GPUs.

```
alien-cli -i input.sim -o output.sim -t 100000
```

| Option | Meaning |
| --- | --- |
| -i | The simulation file to run. |
| -o | The file to which the result is written. |
| -t | The number of time steps to calculate. Without it, the simulation runs until you press Q or Ctrl+C, which also writes the output file. |
| -u, -p | User name and password for the ALIEN server. |
| --upload-name, --upload-interval | Upload the running simulation periodically to your private workspace, under the given name followed by a number, every given number of minutes. |
| -d | Debug mode, which runs slower but writes detailed timing and trace files. |
| --plain | Plain text output without colors and status panel. |
| --no-count-as-active | Do not count this run as an active simulator, see below. |

While a simulation runs, *alien-cli* counts as an active simulator in the **Browser** of the graphical program, also without a login. It reports to the server every 5 minutes, unless --no-count-as-active is given or **Count me as active** is disabled in **Settings > Network settings** of the graphical program for the same user account.

A Docker image with the command line interface is published as *chrxh/alien:nightly*, see the README of the project for details.

The **console mode** of the graphical program (Alt+C) offers a similar experience without leaving the user interface for good, see [The user interface](user-interface.md#console-mode).

## Log files

ALIEN writes its messages to *log.txt* in its folder. The window **Log** (Alt+7) shows the same messages. If ALIEN crashes, enable **Settings > Debug mode**, reproduce the problem and attach *log.txt* and the trace file to a bug report, see [Troubleshooting](troubleshooting.md).
