# Release notes

## [3.1.1] - 2022-04-25
### Added
- new simulation parameters: cellFunctionMinInvocations, cellFunctionInvocationDecayProb, cellFunctionWeaponConnectionsMismatchPenalty and cellFunctionWeaponTokenPenalty
- new simulation example: Maze.sim

### Changed

### Removed

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


