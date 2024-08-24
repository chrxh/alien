# Release notes

## [4.10.1] - 2024-08-24
### Deleted
- engine, gui/CUDA settings: input of threads per block removed (internal routines automatically determine a good value)

### Fixed
- gui/simulation view: performance of rendering with high zoom level improved
- serialization: incorrect conversion of mutation rates into new genome copy mutations when loading simulations with versions below 4.10 fixed
- gui/simulation parameters: deletion of parameter zones led to the unwanted deletion of other zones

## [4.10.0] - 2024-08-17
### Added
- engine: sensor and reconnector cells can be restricted to only sensing/connecting to certain type of mutants (same, other, nutrient, handcrafted, less and more complex)
- engine: sensor cells can tag detected cells for attacking if parameter 'Sensor detection factor' is above 0
- engine: sensors can not penetrate handcrafted structures when restricting to scanning for certain mutants
- engine: possibility to force muscle cells to obtain movement angles from nearby sensors (+ simulation parameter under addon 'Advanced muscle control')
- engine: possibility to redefine max age of permanently inactive or of emergent cells (+ simulation parameters under addon 'Cell age limiter')
- engine: possibility to reset age after cell switches from 'Under construction' to 'Ready' (+ simulation parameter under addon 'Cell age limiter')
- engine: allow to disable radiation sources in spots
- gui/statistics: diversity and average genome complexity plots added
- gui/statistics: throughput statistics
- gui/simulation view: attack visualization (+ simulation parameter)
- gui/simulation view: cell glow (+ simulation parameter under new addon 'Cell glow')
- gui/simulation view: configurable cell radius via simulation parameter
- gui/mass operation, engine: mass operation for randomizing mutation ids

### Changed
- engine: replace continuous mutation rates (which are applied in each time step) by genome copy mutations (which are only applied when genomes are copied)
- engine: make certain mutation rates (all except for translation, duplication and color mutations) dependent on the genome size
- engine: avoid that creature is able to eat its offspring when it is currently under construction
- engine: parameters 'Cell max force' and 'Maximum distance' are now color-dependent

### Fixed
- gui/simulation view: visibility of cells with low energy increased
- gui/browser: apply text filter to all workspaces and resource types
- serialization: load/save real-time counter in autosave
- engine: avoid genome bloating (with separated parts) due to mutations 
- engine: incorrect construction processes based on single cell genomes with multiple branches and/or repetitions fixed

## [4.9.1] - 2024-04-26
### Added
- gui/browser: label new simulations
- gui/browser: allow to replace simulations and genomes

### Changed
- gui/browser: ignore upper and lower case in the browser filter

### Fixed
- gui/browser: preserve subfolder names when renaming folders

## [4.9.0] - 2024-04-22
### Added
- engine, gui/simulation parameters: conditional and unconditional energy inflow from external energy source
- engine, gui/simulation parameters: energy backflow to the external energy source
- gui/simulation view: combined mutation and cell function coloring
- gui/temporal control window: real-time counter

### Changed
- gui/simulation view: cells are rendered smoother
- engine, gui/simulation parameters: external energy becomes a scalar value (no color dependence)

### Deleted
- engine, gui/simulation parameters: energy pump from constructors (substituted by conditional energy inflow)

### Fixed
- serialization: activation of necessary add-ons for old simulation parameter files fixed
- gui/browser: preserve subfolders when renaming folders

## [4.8.2] - 2024-04-05
### Changed
- engine, genome editor: multiple construction flag is replaced by number of constructions
- engine: if construction process failed, destruction is initiated (via dying cell state)
- engine: gap in the completeness check for multiple constructions closed by taking the number of constructions into account

### Fixed
- serialization: current position in the genome for constructors was not loaded correctly

## [4.8.1] - 2024-03-16
### Changed
- engine: the dying state of cells can spread to cells under construction even if they belong to other an other creature

### Fixed
- engine: construction with infinite repetitions fixed
- gui: selection of large sections in edit mode fixed

## [4.8.0] - 2024-02-25
### Added
- gui/simulation view: borderless rendering (world is rendered periodically) + parameter to (de)activate
- gui/simulation view: adaptive space grid for orientation + parameter to (de)activate
- gui/simulation view: new coloring which highlights particular cell functions
- gui/simulation parameters: addon widget introduced
- gui/simulation parameters: center button in spatial control
- engine, gui/simulation parameters: attacker and particle absorption depending on genome complexity
- engine, gui/simulation parameters: addon for genome complexity measurement
- engine, gui/simulation parameters: low and high velocity penalty for energy particle absorption

### Changed
- gui/simulation parameters: move expert settings for absorption, attacker, external energy and color transition to addons
- gui/simulation view: short white background flash after creating and loading snapshots (now called flashbacks)
- gui/simulation view: cross cursor in edit mode
- gui/simulation view: mutation coloring adapted such that color changes occur only after major structural mutations
- gui/statistics: adapt plot heights depending on the visible data points
- engine: memory consumption reduced (~ 10% depending on the data)
- server: support for large simulation files (up to 144 MB)

### Fixed
- gui/simulation parameters: show base tab when new simulation is loaded
- gui/statistics: maintain previous statistics data after resizing the simulation
- engine: fixed rare and spontaneous crashes that occur when many cells and connections are destroyed
- engine: new completeness check counting the actual cells of the creature against the cells in the genome 
- engine: wrong displacement calculation in case of zooming and moving objects fixed

## [4.7.3] - 2024-01-09
### Fixed
- engine: drop restriction on start construction angle
- engine: truncate neural activities to avoid overflow

## [4.7.2] - 2024-01-06
### Added
- engine, gui/simulation parameters: splitting of energy particles above a certain minimum energy
- engine: force fields also affect energy particles

### Fixed
- gui/browser: beginning and ending whitespaces of folder names are ignored for automatic grouping

## [4.7.1] - 2024-01-05
### Added
- gui/browser: shared symbol in private workspace in front of items to indicate if it is public
- gui/upload dialog: toggle if simulation or genome should be shared in public workspace or not

### Changed
- gui/browser: private workspace also shows items from the public workspace if they belong to the same user

### Fixed
- gui/browser: tooltip for downloading and reacting was broken

## [4.7.0] - 2024-01-04
### Added
- gui/browser: private workspace for each user account added
- gui/browser: move simulations, genomes or folders to other workspace (via toolbar buttons)
- gui/browser: edit simulations, genomes or folders (via toolbar buttons)
- gui/browser: expand and collapse folder content (via toolbar buttons)
- gui/browser: cache for speeding up downloading simulations
- gui/upload dialog: validation of user input to allowed characters
- gui/upload dialog: upload simulation or genome to folder
- engine, gui/simulation parameters: individual cell color mutation

### Changed
- gui/browser: layout (in particular, new widget for selecting workspace)
- engine: restrict the fusion of energy particles to certain energies

## [4.6.0] - 2023-12-29
### Added
- gui/browser: support for displaying folders and subfolders
- gui/browser: folders for simulations and genomes are automatically created by parsing their names for `/`
- gui/browser: allow uploading to a selected folder
- gui/browser: show number of simulations per folder

### Changed
- gui/browser: tree view instead of a pure tabular view
- gui/browser: simulations and genomes can be selected for user actions (e.g. deletion)

### Removed
- gui/browser: column for actions removed

## [4.5.1] - 2023-12-16
### Added
- new simulation parameters to reduce energy particle absorption for cells with fewer connections
- new simulation parameters to configure the decay probability of dying cells
- validation of normal and minimal cell energy parameters in the simulation parameter window

### Changed
- parameter 'Genome size bonus' only takes into account the non-separating parts of the genome

### Fixed
- layout problems with the display of color matrices fixed

## [4.5.0] - 2023-12-09
### Added
- engine, gui: entire statistical history are saved along with a simulation
- gui/statistics: configurable plot heights
- gui/statistics: collapse specific plots
- gui/browser: statistical history are up/downloaded along with a simulation
- gui/sim view: draw boundaries of the world
- Python script `FindFurtunateTimeline`: constantly monitors populations in a simulation and resort to savepoints in case of extinction

### Changed
- gui/statistics: widgets for real time and time horizon parameter are aligned with the other widgets

### Removed
- export statistics function

### Fixed
- in simulation runs via CLI, certain simulation parameters are adjusted as in the GUI (e.g. if the max age balancer is switched on, external energy consumption)
- wrong color conversion HSV -> RGB fixed (relevant for mutation and genome size coloring)

## [4.4.3] - 2023-11-29
### Added
- show text background when rendering is disabled

### Fixed
- fixed insertion mutation behavior that led to undesirably high repetitions and made this mutation type more or less useless

## [4.4.2] - 2023-11-25
### Added
- external energy source, which is controllable by two new parameters

### Removed
- parameter `unlimited energy for constructors` removed

## [4.4.1] - 2023-11-15
### Added
- allow deletion of selection via DEL key

### Fixed
- allow copy & paste when inspection windows are open
- minor layout correction in sliders and genome widgets

## [4.4.0] - 2023-11-08
### Added
- engine, gui: genomes and sub-genomes contain repetition information and concatenation angles
- engine, gui: support for infinite repetitions of genome structures
- engine, gui: reconnector cells (can form and break bonds depending on neural activities)
- engine, gui: detonator cells (can detonate depending on neural activities)
- engine, gui: neuron cells extended: 5 different activation functions can be selected for each neuron
- engine, gui: simulation parameters for reconnectors and detonators
- gui/sim view: 2 new coloring available: "Cell state" and "Genome size"
- gui/genome preview: markers for start, end, infinity repetition, multiple construction and self-replication
- gui/genome preview: visualization optimized depending on the zoom level
- gui/genome editor: mass operation for changing colors of cells optionally including sub-genomes
- gui/neuron editor: reset, set identity and randomize function
- gui/browser: button to open Discord server
- gui/statistics: plots and exports for reconnections and detonator events
- logging: more log messages during startup

### Changed
- engine: restrict cluster decay on cells which belong to the same creature 
- engine: allow insert mutations on empty genomes
- gui: toolbar buttons in creator and multiplier windows are made selectable
- gui/inspection: show ids (cell, creature, mutation) in base tab
- gui/genome editor: icons for expanding and collapsing changed
- gui/sim parameter: focus base tab when opening new simulation with different spots

### Fixed
- show correct tab when sub-genome is edited
- completeness check evaluates creatureIds of the cells in order to determinate the creature's boundaries
- prevent crash in case that a single genome exceeds 8 KB
- genome editor layout bug fixed when separator is moved out of range 
- layout problems after resizing in several dialogs fixed (e.g. in display, gpu, network settings)
- invalid zooming prevented

## [4.3.0] - 2023-09-23
### Added
- gui/browser: tab widget added to show the uploaded genomes and simulations from server
- gui/browser: possibility to upload and download genomes
- gui/genome editor: toolbar button added to upload current genome
- cli: file logger added (creates log.txt)

### Fixed
- gui/browser: layout problem for multiline descriptions

## [4.2.0] - 2023-09-21
### Added
- command-line interface for running simulation files for a specified number of time steps
- statistics can be exported with CLI

### Fixed
- csv-file in statistics export corrected

## [4.1.1] - 2023-09-16
### Added
- show confirmation dialog for deleting a simulation

### Changed
- hide trash icon for simulations from other users

## [4.1.0] - 2023-09-07
### Added
- gui/browser: user can open an emoji window in order to react with various emoji types
- gui/browser: list of emojis and counts are shown for each simulation entry
- gui/browser: clicking on an other user's emoji adds same reaction
- global: debug mode added that provides more accurate error messages and can be activated with the '-debug' parameter

### Changed
- gui/browser: different colors for the action buttons

## [4.0.2] - 2023-09-03
### Added
- gui/browser: show online since last day status
- gui/browser: show number of simulators

### Changed
- engine: injection mode 'Cells under construction' is replaced by 'Only empty cells'
- engine: scanner cells return data (angel, distance) of last match if no match was found

## [4.0.0] - 2023-08-30
### Added
- engine: support for dynamic simulation parameter zones, barriers and particle sources: they can move with a fixed velocity
- engine: different cell colorings: according to mutants, according to standard cell color, none
- engine: different shapes for radiation sources
- engine: geometry presets for phenotypes
- engine: mutation types added: uniform color mutation and geometry mutation
- engine: energy pump system for constructors
- engine: attacker cells can be configured to attack only cells already targeted by nearby sensors
- engine: attacker strength can be dependent on the size of its genome to which it belongs
- engine: same mutants can be protected by each others attacks
- engine: muscle cells can directly control the relative direction of movements via their activities
- engine: balancing of maximum age per color depending on the population size
- gui/statistics: additional plots for self-replicators, viruses and total energy
- gui/browser: user list added: showing online status, gpu model (if approved), time spent, stars
- gui/browser: toggle 'community creation' in simulation browser
- gui/browser: version validation for simulation files
- gui/help: getting started window supplemented by additional sections (basic notions, examples, simulation parameters, editing tools, FAQ)
- gui/view: mouse wheel support for zooming
- gui/view: automatically scale all window sizes according to OS content scale
- gui/genome editor: support for geometry presets and allow modify angles and connections
- gui/mass operations: coloring cells in genomes
- gui/simulation parameters, genome editor, inspection: tooltips for almost all fields added
- gui/simulation parameters: parameters for configuring new features (attacker can destroy other cells, sensor targeting for attackers, same mutant protection, genome size bonus, ...)

### Changed
- engine: more realistic collision algorithm between cells and barriers
- engine: insertion mutation treats inserts to sub-genomes equally likely
- gui/global: if OS=Windows: settings are saved to the Windows registry
- gui/view: better zooming experience: make continuous zoom speed of the computational workload
- gui/browser: all example are available as non-community creations in the browser
- gui/inspection: genome tab extended and new layout used
- new startup simulation featuring sensor-equipped consumer and plant ecosystem

### Removed
- simulation files in folder 'examples' removed (they are available in the sim browser instead)
- simulation parameter 'Same color energy distribution' for attackers removed

### Fixed
- display an error message if GPU memory allocation failed and allow to continue
- fixed wrong parameter calculation in case of overlapping parameter zones
- plots for accumulated and averaged values have been corrected
- loopholes and timeout bug for completeness check fixed
- fetching simulation list optimized
- unwanted conversion to lower case in input fields removed

## [4.0.0-beta] - 2023-04-25
### Added
- engine: new cell functions and corresponding parameters: neurons, transmitters, nerves, injectors and defenders
- engine: neural activity for cells
- engine: radiation sources and extended logic (absorption factors, cell age radiation, high energy radiation)
- engine: simulation parameter override function for spots
- engine: most simulation parameters can be configured by cell color
- engine: SPH solver
- engine: stiffness per cell
- engine: linear and central force fields
- engine: new mutation types
- engine: living states for cells
- engine: tracking of genome generation
- engine: energy pump function for constructors
- engine: cell rendering improved
- gui/genome editor: editor with preview added
- gui/pattern editor: inspect genome function added
- gui/mass operations: dialog added
- gui/simulation parameters: save, load, copy and paste function
- gui/statistics: plot each cell function activity
- gui/statistics: every plot can be broken down by colors
- gui/statistics: plot values in the long-term view are smoothed at regular intervals so that they remain readable
- gui/statistics: histogram for cell ages
- gui/browser: version check for simulation files
- gui/browser: filter for community creations
- gui/creator: pencil width for brush draw function and fitting mouse cursor
- various examples

### Changed
- cell functions obtain input from and provide output to neural activities instead from/to token memories
- constructor cells contain a construction sequence for an entire cell cluster (encoded in a genome) instead of performing a single cell construction
- 'token branch number' to 'execution order number' changed

### Removed
- tokens
- cell function for computing operations
- cell code editor and compiler
- cell memory editor
- symbol map and symbol editor
- time-varying simulation parameters

## [3.3.1] - 2023-02-03
### Added
- allow comments starting with # in cell code
- decompiled cell code shows matching symbols in comments

## [3.3.0] - 2022-10-05
### Added
- extended color semantic for cells: food chain color matrix and cell color transition rules
- new simulation parameters for cell colors
- shader parameter window
- Symbiosis examples

### Fixed
- process statistics corrected (showing processes per time step)
- deadlock problem during removing cells fixed
- precision of simulation parameters increased (relevant for mutation rates)

## [3.2.3] - 2022-07-31
### Added
- toolbar for browser window: refresh, login, logout, upload

## [3.2.2] - 2022-07-24
### Added
- show downloads in browser
- more spore examples

### Changed
- allow scanning of token blocked cells

## [3.2.1] - 2022-07-13
### Added
- Mycelial Networks example

### Changed
- calculation of inner forces improved: prevent unwanted rotations and movements
- performance optimizations in case of many cell connections

## [3.2.0] - 2022-06-25
### Added
- browser for downloading simulations from a server
- registration of users on a server
- upload simulations
- rate simulations by likes
- image to pattern converter
- new examples

### Changed
- window transparency lowered
- simulation parameter group renamed

### Fixed
- editor performance increased through caching
- scanner function scans correct cell

## [3.1.2] - 2022-05-01
### Fixed
- rigidity spot parameter is saved correctly

## [3.1.1] - 2022-04-25
### Added
- new simulation parameters: cellFunctionMinInvocations, cellFunctionInvocationDecayProb, cellFunctionWeaponConnectionsMismatchPenalty and cellFunctionWeaponTokenPenalty
- new simulation example: Maze.sim

### Fixed
- keep metadata after copy & paste of patterns
- replicators in Dark Forst.sim repaired
- forbid digestion of barrier cells
- crystals in Living Crystals.sim are stabilized
- color code 6 (gray) in pattern editor is remember like the other colors

## [3.1.0] - 2022-04-18
### Added
- barrier cells introduced: indestructible and immobile
- checkbox for setting/unsetting barriers in pattern editor, creator and inspection window
- performance notice in Getting started window
- new examples which use barriers: Soft Balls.sim and Multiverse.sim

### Changed
- file format contains program version (compatibility with old format remains!)

### Fixed
- autosave bug fixed (led to long lags every 20minutes)

## [3.0.0] - 2022-04-09
### Added
- new engine with soft-body dynamics
- new GPU-based editor (enables editing at all zoom levels)
- new simulation and pattern examples 
- more modern looking user interface with imgui
- sub-windows to group all simulator functions (temporal control, spatial control, simulation parameters etc.)
- cross-platform support
- spatially different simulation parameters and background colors via spots
- flow generator
- pin edit windows to multiple cells
- statistics include cell processes and number of cells by colors
- freehand drawings
- display resolution and frame rate settings
- automatic change of GPU array sizes for entities
- compression of simulation files
- system font size scaling is taken into account
- enable/disable user interface

### Changed
- simulation parameters adapted to new engine

### Removed
- settings for array sizes
- main window toolbar
- bug reporting after crash
- cell function for communication

## [2.5.3] - 2021-06-12
### Changed
- more specific error message in case of cudaErrorUnsupportedPtxVersion
- removed useless size info at the beginning in saved files (sim/parameters/...)

### Fixed
- wrong view port size bug fixed
- integration tests repaired

## [2.5.2] - 2021-05-27
### Added
- planet gaia example files

### Fixed
- reduction of memory reservations in editor to prevent out-of-memory exceptions
- prevent invalid-map-key exceptions in cases where the selection is no longer available in the editor
- display more precise information in case of exceptions
- layout in dialogs improved

## [2.5.1] - 2021-05-21
### Fixed
- important memory leak fixed
- destruction of cell clusters corrected: looks now much nicer!
- allow only full screen because of a Qt bug

## [2.5.0] - 2021-05-19
### Added
- navigation mode enables continuous zooming
- new editing function: modifier key (CTRL) allows the precise selection of cells
- new editing function: randomize cell function
- new editing function: automatic generation of token branch numbers on cell cluster
- new editing function: remove unused cell connections
- selection of the graphics card with the highest compute capability, if more than one is found
- checking for newer versions
- display in the infobar when time steps per second are restricted

### Changed
- better rendering performance due to OpenGL-CUDA interoperability
- better image quality for low zoom factors
- motion blur filter
- more intense glow filter
- cells are represented in the vector view by a circle with a color gradient
- two toolbar buttons (instead of one) for switching between editor and pixel/vector view
- using flat design for main window
- more beautiful progress bar
- activated icons in the toolbar glow
- simulation parameters renamed
- meaningful error message in case the system requirements are not met
- colors in editor more balanced
- infobar is disabled by default
- getting started info revised
- logo reduced in size
- startup example revised
- collision example revised

### Fixed
- prevent exception on closing

## [2.4.7] - 2021-04-14
### Changed
- error messages improved (call stack and bad alloc message)

## [2.4.6] - 2021-04-11
### Changed
- created cell have maximum bonds corresponding to simulation parameters

## [2.4.5] - 2021-04-08
### Fixed
- fixed exception on closing
- negative energies prevented

## [2.4.4] - 2021-04-04
### Fixed
- fixed crash when minimum hardware requirements are not met

## [2.4.3] - 2021-04-02
### Added
- new simulation parameter "offspring token suppress memory copy" (default=false)

### Fixed
- font in metadata tab from visual editor corrected

## [2.4.2] - 2021-03-24
### Changed
- selected cells and particles have lighter colors and smaller sizes in the editor
- border colors of cells and particles darkened
- unused simulation parameter "cluster properties -> max size" removed

## [2.4.1] - 2021-03-17
### Fixed
- fix performance bug when using glow effect

## [2.4.0] - 2021-03-05
### Added
- disc-shaped structures can be created
- selections can be colored

### Changed
- simulation parameters and symbol maps are saved in JSON format
- saved simulations are divided into 4 files: *.sim, *.settings.json, *.parameters.json and *.symbols.json

### Fixed
- second scrollbar in token editor avoided

## [2.3.0] - 2021-02-21
### Added
- infobar, which displays general information, monitor information and logging protocol
- show bug report dialog after crash that allows to send reports to a server

### Changed
- using standard font in dialogs and most widgets
- requirements of the examples lowered

### Fixed
- fix runtime problems for cells with constructor function

## [2.2.2] - 2021-02-10
### Changed
- upgrade to CUDA 11.2, Qt 6.0.1, boost 1.75.0 and VS 2019
- parallel console window is not opened
- examples renamed

### Fixed
- timeout problem during simulation run
- fix crash when monitor is open and simulation or general settings are changed 
- fix crash after inserting collection

## [2.2.1] - 2021-02-05
### Fixed
- fixed crash at display resolutions higher than Full HD

## [2.2.0] - 2021-02-03
### Added
- vector graphic view mode is active from zoom level 4 onwards
- startup screen shows version
- validation of values in "General settings" and "New simulation" dialog
- examples (simulation and collections)

### Changed
- info banner shows rendering mode
- "Computation settings" dialog restructured and renamed to "General settings"
- startup simulation changed

### Removed
- experimental feature "Web access" disabled (will be enabled when finished)

### Fixed
- showing messages in case of CUDA errors instead of immediate termination
- fix typos

## [2.1.0] - 2021-01-17
### Added
- usability: showing first steps window after startup
- allow moving objects with the mouse pointer
- ability to connect to a server and send simulation real-time data (still experimental)
- automatic switching between editor and pixel view depending on zoom level
- new examples

### Changed
- nicer and larger toolbar icons
- new logo
- disable editor for low zoom levels

### Fixed
- fix empty simulation after "step backward" or "restore snapshot"
- fix center scroll position after startup


