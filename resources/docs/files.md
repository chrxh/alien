# Files and command line

This chapter describes the files ALIEN works with, how simulations are saved automatically, how to share your work through the browser and how to run simulations without a window.

## File types

| File | Contents |
| --- | --- |
| *.sim* | A complete simulation: all objects, energy particles, creatures and genomes, the simulation parameters, the statistics history, the time step and the position of the view. |
| *.genome* | A single genome, as created in the genome editor. |
| *.settings.json* | A set of simulation parameters including layers and radiation sources. It can be loaded into any simulation. |
| *.png* | A picture created with **Simulation > Save picture**. **Tools > Image converter** (Alt+Y) turns a PNG image into a pattern of solid objects. |
| *.mp4* | A video recorded with **Tools > Recorder**. |
| *.flight.json* | The waypoints and settings of a camera flight, as created in **Tools > Recorder**. |

## Opening and saving

- **Simulation > Open** (Ctrl+O) loads a simulation file and **Simulation > Save** (Ctrl+S) saves the current simulation.
- **Simulation > New** (Ctrl+N) creates an empty world.
- Genomes are opened and saved in the tool bar of the genome editor.
- Parameter sets are opened and saved in the tool bar of the **Simulation parameters** window.
- Camera flights are opened and saved in the tool bar of the **Recorder** window.
- If **Settings > Save on exit** is enabled, which is the default, ALIEN saves the current simulation to *resources/autosave/autosave.sim* in the ALIEN folder when it closes. ALIEN opens this file at every start until you delete it, even after **Save on exit** has been disabled.

## Autosave and save points

The **Autosave** window (Alt+5) protects long runs. It can create save points of the current simulation at regular intervals. These automatic save points are switched off by default. To switch them on, open the **Settings** section of the window and tick the checkbox in front of **Autosave interval (min)**.

- **Autosave interval (min)** sets the time between two save points, 40 minutes by default. The interval counts minutes of real time, so save points are also created while the simulation is paused.
- **Directory** sets where the save points are stored.
- **Mode** is *Limited save files* by default. In this mode, **Number of files** limits how many save points are kept, 20 by default. Each new save point, including one created by hand, deletes the oldest ones beyond this number. In *Unlimited save files* mode, all save points are kept.
- The buttons in the tool bar create a save point by hand or delete save points. The download button in a row loads that save point.

## Sharing in the browser

The browser (Alt+B) connects ALIEN with the ALIEN server, where users share simulations and genomes.

- **Browsing** works without an account. The **Featured** workspace contains simulations of the ALIEN project, and the **Community** workspace contains everything users have shared.
- **Logging in.** Choose **Network > Login** (Alt+L) or click the account chip in the browser. The login dialog also lets you create a new account. With an account you get a **Private** workspace and can upload your own work.
- **Uploading.** **Network > Upload simulation** (Alt+D) uploads the current simulation, and **Network > Upload genome** (Alt+Q) uploads the genome of the genome editor. The name is prefilled with your user name as a folder, followed by the project name of the simulation or the name of the genome, for example *UserName/Ocean*. A slash in the name creates folders, for example *My worlds/Ocean*. You can add a description.
- **Visibility.** **Share with the community** makes the upload visible to everyone in the **Community** workspace and also posts it on the ALIEN Discord server. **Keep private** shows it only in your **Private** workspace. After the upload, **Show in browser** selects the new entry in the browser.
- **Managing.** The browser tool bar lets you rename your uploads, replace them with a newer version, switch them between private and community and delete them.
- **Reacting.** No account is needed. Click the heart of an entry to give it a heart and click it again to take it back. A right click chooses another reaction. **Liked** below the entries shows only the entries you reacted to. The gallery view can sort by reactions, date or downloads.
- **Credits.** While a simulation from the browser is open, the right end of the menu bar shows its name, its author and a heart. Click the name to show the entry in the browser or the heart to give it a heart.

## Command line interface

ALIEN includes a second program, *alien-cli* (on Windows *alien-cli.exe* for NVIDIA and *alien-cli-amd.exe* for AMD), which runs a simulation without any window. It is useful for long experiments, for automation and on computers without a monitor, such as rented cloud servers with a GPU.

```
alien-cli -i input.sim -o output.sim -t 100000
```

| Option | Meaning |
| --- | --- |
| -i | The simulation file to run. |
| -o | The file to which the result is written. |
| -t | The number of time steps to calculate. Without it, the simulation runs until you press **Q** or **Ctrl+C**. Both stop the run and write the output file. |
| -u, -p | User name and password for the ALIEN server. The long forms are --user and --password. |
| --upload-name | The base name for periodic uploads of the running simulation to your **Private** workspace. Each upload gets this name followed by an underscore and a consecutive number, for example *Run_1* and *Run_2*. Requires --upload-interval and a login with -u and -p. |
| --upload-interval | The number of minutes between two uploads. The first upload takes place one interval after the start. Requires --upload-name. |
| -d | Debug mode, which runs slower but writes detailed timing and trace files. The long form is --debug. |
| --plain | Plain text output without colors or a status panel. |
| --no-count-as-active | Do not count this run as an active simulator (see below). |
| -h | Show all options and exit. The long form is --help. |

Like the graphical program, *alien-cli* is included in the number of active simulators that the **Browser** window shows, even without a login. While it runs, it reports to the ALIEN server every 5 minutes, the first time 5 minutes after its start. Each instance stays counted for 25 minutes after its last report. Instances that share one IP address count once. To leave a run out, start it with --no-count-as-active or disable **Count me as active** in **Settings > Network settings** of the graphical program. Both programs share this setting if they run on the same computer under the same operating system user, regardless of the ALIEN account.

A Docker image with the command line interface is published as *chrxh/alien:nightly*. See the README of the project for details.

The **Console mode** of the graphical program (Alt+C) is similar to *alien-cli*. It hides the user interface and switches off the rendering until you press **Esc**. See [The user interface](user-interface.md#console-mode).

## Log files

ALIEN writes its messages to *log.txt* in the ALIEN folder and replaces this file at every start. The **Log** window (Alt+7) shows the same messages. If ALIEN crashes, enable **Settings > Debug mode**, reproduce the problem and copy *log.txt* and *trace.txt* from the ALIEN folder before you start ALIEN again. Attach the copies to a bug report. If ALIEN crashes before you can use the menu, start it from the command line with the option -d, which switches on debug mode from the beginning. See [Troubleshooting](troubleshooting.md) for details.
