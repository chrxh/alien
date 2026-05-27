# ALIEN - Artificial Life Environment

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Project Overview
ALIEN is an artificial life simulation tool based on a specialized 2D particle engine in CUDA for soft bodies and fluids. Each simulated body consists of a network of particles that can be upgraded with higher-level functions, ranging from pure information processing capabilities to physical equipment (such as sensors, muscles, weapons, constructors, etc.) whose executions are orchestrated by neural networks. The bodies can be thought of as agents or digital organisms operating in a common environment.

## Security Configuration

### CodeQL Security Checks
**DO NOT use CodeQL security checks on this repository.** The repository is too large for CodeQL analysis to complete successfully. 

- Do NOT run the `codeql_checker` tool
- Do NOT attempt any CodeQL security scanning
- Skip all security vulnerability scanning steps that involve CodeQL

### Git Submodule Configuration
**DO NOT update the vcpkg submodule in external/vcpkg.** The vcpkg version is pinned to a specific commit for stability.

- Do NOT run `git submodule update` commands
- Do NOT modify the external/vcpkg submodule pointer
- Do NOT commit changes to external/vcpkg
- If you see external/vcpkg changes in `git status`, use `git restore external/vcpkg` to discard them
- The vcpkg submodule is managed separately by repository maintainers

## Working Effectively

### Bootstrap, Build, and Test the Repository
**NEVER CANCEL long-running commands. Build may take 4-5 minutes. Use long timeouts.**

```bash
cmake --build build --config Release -j32
```

### Run Tests
```bash
cd build

# Tests
./EngineInterfaceTests  # <1 second
./NetworkTests          # <1 second
./PersisterTests        # ~1.4 seconds
./EngineTests           # >4 min

# Debug mode for EngineTests (use -d for precise kernel failure info)
./EngineTests -d --gtest_filter=TestSuite.TestName  # Much slower, use for debugging specific tests only
```

**Note on `-d` flag**: The `-d` parameter enables debug mode which synchronizes CUDA after each kernel call. This provides more precise information about which kernel is failing, but significantly increases test execution time. Only recommended when debugging specific failing tests, not for full test suite runs.

### Run the Applications
```bash
cd build

# GUI application (requires X11 display and NVIDIA GPU)
./alien

# Command-line interface (requires .sim and .settings.json files)
./cli --help
./cli -i input.sim -o output.sim -t 1000
```

## Validation

## Validation

### Build Validation
- **Build succeeds** and produces all expected executables: `alien`, `cli`, `EngineTests`, `EngineInterfaceTests`, `NetworkTests`, `PersisterTests`
- **Build time**: ~1 min on 32-core system with `-j32` parallelization
- **Clean configuration**: ~6 seconds
- **No build errors or warnings** when following the exact commands above

### Test Validation  
- **EngineInterfaceTests**: tests pass in <1 second
- **NetworkTests**: tests pass in <1 second
- **PersisterTests**: tests pass in ~1.4 seconds
- **EngineTests**: tests pass in >4 min
- Pure GUI-only changes in `source/Gui/` that do not affect engine, network, persistence, or CLI logic do **not** require running tests
- For all other changes, run `./EngineInterfaceTests && ./NetworkTests && ./PersisterTests && ./EngineTests` to verify your changes don't break core functionality
- `./EngineTests` are most important and contain the entire simulation logic written in CUDA

### Application Validation
- **CLI works**: `./cli --help` shows usage information
- **GUI requires**: NVIDIA GPU + X11 display (cannot run in headless CI environments)
- **GUI error handling**: Provides clear error messages when GPU requirements aren't met
- **Simulation files**: CLI expects `.sim` and `.settings.json` files (not included in main repo)

### Validation Workflow for Changes
```bash
# Always run this validation sequence after making code changes:

# 1. Format code (if modified)
clang-format --style=file:source/_clang-format -i path/to/modified/files.cpp

# 2. Build (NEVER CANCEL - ~ 1 minute)
cmake --build build --config Release -j32

# 3. Run core tests (required unless the change is pure GUI-only in source/Gui)
cd build && ./EngineInterfaceTests && ./NetworkTests && ./PersisterTests & ./EngineTests

# 4. Test CLI functionality
./cli --help
```

### Code Formatting
```bash
# Format code using project's clang-format config
clang-format --style=file:source/_clang-format --dry-run --Werror source/path/to/file.cpp

# Format all files (if needed)
find source -name "*.cpp" -o -name "*.h" | xargs clang-format --style=file:source/_clang-format -i
```

## Technology Stack & Requirements
- **Languages**: C++23, CUDA 20, Python (for CLI tools)
- **Build System**: CMake 3.31+, vcpkg package manager
- **GPU Computing**: NVIDIA CUDA (requires compute capability 7.5+)
- **GUI Framework**: Dear ImGui with custom widgets
- **Dependencies**: All managed via vcpkg manifest mode (`vcpkg.json`)

## Repository Structure & Navigation
- `source/`: Main C++ and CUDA source code
  - `source/Base/`: Common utilities, math, logging
  - `source/Cli/`: Command-line interface
  - `source/EngineGpuKernels/`: CUDA kernels for simulation
  - `source/EngineImpl/`: CPU-side engine implementation
  - `source/EngineInterface/`: Abstract simulation APIs
  - `source/EngineInterfaceTests/`: Unit tests for EngineInterface
  - `source/EngineTests/`: Integration tests for engine
  - `source/Gui/`: ImGui-based user interface
  - `source/Network/`: HTTP client for cloud features
  - `source/PersisterImpl/`: File I/O and serialization
  - `source/_clang-format`: Code formatting configuration
- `external/`: Third-party libraries (vcpkg submodule)
- `resources/`: Runtime assets (shaders, fonts, icons)
- `scripts/CLI-Tools/`: Python automation tools
- `vcpkg.json`: Dependency manifest for package management

## Common Development Tasks

### InspectionWindow Tree Node Behavior
The InspectionWindow uses tree nodes to display object/cell/particle properties. The tree node open/close states follow these rules:
- **First open ever**: Only the "Object" (or "Energy particle") node is fully expanded (including children). All other nodes (e.g., "Cell", "Creature", "Signals", etc.) are collapsed via `defaultOpen(false)`.
- **Subsequent opens** (same or different object): The last tree node states are restored. If a tree node name matches a previously opened node, it retains its last open/close state. Otherwise, it defaults to the `defaultOpen` parameter.
- **Scroll position**: The scroll position is also persisted and restored across window reopens.

Implementation: `AlienGui::BeginTreeNode` generically saves/restores each tree node's open/close state by name via the static `_savedTreeNodeStatesByName` map. On first encounter of a new ImGui ID, it applies any previously saved state; otherwise it uses the `defaultOpen` parameter.

### Command-Line Interface (CLI)
The project includes a CLI for headless simulation execution:
```bash
# Basic simulation run (requires existing .sim and .settings.json files)
./cli -i example.sim -o output.sim -t 1000

# The CLI generates three outputs:
Python automation tools are available in `scripts/CLI-Tools/`:
- `FindFortunateTimeline.py`: Automated simulation with savepoints and rollback logic

### Code Style & Formatting
- **Formatting config**: `source/_clang-format` file contains project style rules
- **Line length**: 160 characters maximum
- **Indentation**: 4 spaces, no tabs
- **Braces**: Allman style (opening brace on new line)
- **Comments**: Avoid unnecessary comments - code should be self-documenting
- **Vector access**: Always use `.at()` instead of `[]` for `std::vector` element access to enable bounds checking

### Naming Conventions
- **Classes**: PascalCase (`SimulationFacade`)
- **Variables/functions**: camelCase (`calculateEnergy`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_PARTICLES`)
- **Files**: `.h` for C++ headers, `.cuh` for CUDA headers, `.cpp/.cu` for implementation

### Architecture Overview
The engine follows a layered architecture:
- **Interface Layer** (`EngineInterface/`): Abstract APIs for simulation operations
- **Implementation Layer** (`EngineImpl/`): CPU-side coordination and data management
- **GPU Compute Layer** (`EngineGpuKernels/`): CUDA kernels for parallel simulation
- **GUI Layer** (`Gui/`): User interface built on Dear ImGui

### Testing Guidelines
- **Test naming**: `*Tests.cpp` files, descriptive test method names
- **Test types**: Unit tests (preferred), integration tests, performance tests
- **GPU tests**: Require NVIDIA hardware, will fail in CI without GPU
- **Always run**: `./EngineInterfaceTests && ./NetworkTests && ./PersisterTests` to verify core functionality

### Performance & Debugging
- **CUDA debugging**: Use `cuda-gdb` or Nsight Compute for kernel debugging
- **Memory profiling**: Use CUDA memory checker for leak detection
- **Build issues**: Clean vcpkg cache and rebuild dependencies if needed
- **GPU requirements**: Compute capability 6.0+ required for all GPU functionality

## External Resources
- [Project Documentation](https://alien-project.gitbook.io/docs)
- [Architecture Overview](https://alien-project.gitbook.io/docs/under-the-hood)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Dear ImGui Documentation](https://github.com/ocornut/imgui)
