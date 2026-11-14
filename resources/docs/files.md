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

The browser (Alt+B) shows the simulations of this computer next to those of the ALIEN server. Its left column lists the workspaces. **My simulations** and **Autosaves** are on this computer, while **Featured**, **Community** and **My uploads** are on the server. **Liked** collects the entries of the server you reacted to. On the **Genomes** tab, **My genomes** takes the place of **My simulations**, and there are no autosaves.

- **Saving.** **Simulation > Save** (Ctrl+S) opens the **Save simulation** dialog. It shows a picture of the current view and asks for a name and an optional description. **Save to** chooses the target. *My simulations* is preselected and stores the simulation on this computer. *Community* and *My uploads* upload it to the server, see [Sharing in the browser](#sharing-in-the-browser).
- **Names and folders.** A slash in the name creates folders, for example *My worlds/Ocean*. A simulation that was opened from **My simulations** or saved there before keeps its name as the suggestion. If the name is already in use at the target, the button reads **Replace** and the dialog tells what is replaced.
- **Opening.** In the browser, click a simulation of **My simulations** to see its picture, its description and its data in the right column. Double-click it or choose **Open** there to load it. The right column also renames, shares and deletes the selected simulation.
- **Other locations.** **Simulation > Open** (Ctrl+O) and **Open simulation** in the tool bar of the browser load a simulation file from any folder. *Other location* under **Save to** in the **Save simulation** dialog saves the simulation as a file in any folder.
- **Where the files are.** The simulations of this computer are ordinary *.sim* files in one folder, by default *simulations* in the ALIEN folder. The folders of the browser are its subfolders. The picture of a simulation is a JPG file and its description a text file of the same name next to it. Without a selection, the right column of **My simulations** shows this folder as **Folder**, chooses another one and opens it in the file manager. Simulation files copied into the folder appear in the browser. A simulation without a picture gets one when it is opened.
- **Simulation > New** (Ctrl+N) creates an empty world.
- Genomes are opened and saved in the tool bar of the genome editor. **Save genome** in the tool bar of the browser stores the genome of the genome editor in **My genomes** with a picture and a description, and a double-click on a genome there opens it in the genome editor. The genome files lie in the same folder as the simulations.
- Parameter sets are opened and saved in the tool bar of the **Simulation parameters** window.
- Camera flights are opened and saved in the tool bar of the **Recorder** window.
- If **Settings > Save on exit** is enabled, which is the default, ALIEN saves the current simulation to *resources/autosave/autosave.sim* in the ALIEN folder when it closes. ALIEN opens this file at every start until you delete it, even after **Save on exit** has been disabled.

## Autosave

Autosave protects long runs. It saves the current simulation at regular intervals into the **Autosaves** workspace in the left column of the browser. Autosave is switched off by default. Its settings are in the right column as long as no autosave is selected. To switch it on, tick **Enable autosave**.

- **Folder** is the folder of **My simulations** until you choose another one for the autosaves here.
- **Subfolder** names the folder inside **Folder** in which the autosaves are stored, for example one subfolder for each simulation run. A slash creates nested subfolders.
- **Interval (min)** sets the time between two autosaves, 40 minutes by default. The interval counts minutes of real time, so autosaves are also created while the simulation is paused.
- **Mode** is *Limited save files* by default. In this mode, **Number of files** limits how many autosaves are kept in a folder, 20 by default. Each new autosave deletes the oldest ones of its folder beyond this number. In *Unlimited save files* mode, all autosaves are kept.
- **Autosave now** creates an autosave right away.
- An autosave is named by its time step. Select it to open or delete it in the right column. To keep an autosave for good, open it and save it with **Save** (Ctrl+S) under a name of its own. A click on **Autosaves** in the left column shows the settings again.
- **Earlier versions.** At the first start of this version, the folder in which the Autosave window of an earlier version has created save points becomes the folder of the simulations of this computer. These save points appear in **Autosaves** and are never deleted automatically. Save points that an earlier version has stored directly in the *resources* folder are loaded with **Open simulation** in the tool bar of the browser.

## Sharing in the browser

The browser (Alt+B) connects ALIEN with the ALIEN server, where users share simulations and genomes.

- **Browsing** works without an account. The **Featured** workspace contains simulations of the ALIEN project, and the **Community** workspace contains everything users have shared.
- **Logging in.** Choose **Network > Login** (Alt+L) or click the account chip in the browser. The login dialog also lets you create a new account. With an account you can upload your own work, which **My uploads** lists.
- **Uploading.** Choose *Community* or *My uploads* as the target in the **Save simulation** dialog (Ctrl+S). Alt+D opens the dialog with *Community* preselected. Alt+Q opens the same dialog for the genome of the genome editor, as does **Save genome** in the tool bar of the browser on the **Genomes** tab. With a target on the server, the name is prefilled with your user name as a folder, followed by the project name of the simulation or the name of the genome, for example *UserName/Ocean*. A slash in the name creates folders, for example *My worlds/Ocean*. You can add a description.
- **Sharing a saved simulation.** **Share** in the right column of **My simulations** uploads the selected simulation with its picture, its name and its description. A simulation that carries the name of one of your uploads shows where it has been uploaded and how many reactions it has received.
- **Replacing an upload.** **Replace** in the right column of one of your uploads replaces its content and its picture with the current simulation, or with the genome of the genome editor on the **Genomes** tab. Its name, its description and its reactions are kept.
- **Visibility.** *Community* makes the upload visible to everyone in the **Community** workspace and also posts it on the ALIEN Discord server. *My uploads* keeps it visible only to you. Every upload of yours is listed in **My uploads**, where the right column switches its visibility between *Community* and *Only me*. After the upload, **Show in browser** selects the new entry in the browser.
- **Managing.** Select one of your uploads to rename it, to change its description, to switch it between private and community or to delete it in the right column. To replace an upload with a newer version, save the current simulation under the same name with the same target.
- **Reacting.** No account is needed. Click the heart on the picture of an entry to give it a heart and click it again to take it back. A right click chooses another reaction, and the right column shows all reactions to the selected entry. **Liked** in the left column collects all entries you reacted to. The gallery view can sort by reactions, date or downloads.
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
| --upload-name | The base name for periodic uploads of the running simulation to **My uploads**, visible only to you. Each upload gets this name followed by an underscore and a consecutive number, for example *Run_1* and *Run_2*. Requires --upload-interval and a login with -u and -p. |
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
