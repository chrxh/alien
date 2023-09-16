# Release notes

## [4.1.1] - 2023-09-16
### Added
- show confirmation dialog for deleting a simulation

### Changed
- hide trash icon for simulation from other users

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


