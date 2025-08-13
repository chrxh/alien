# ALIEN - Artificial Life Environment

ALIEN is a C++/CUDA-based artificial life simulation platform with GPU-accelerated physics and evolutionary algorithms. The project consists of multiple executables including a GUI application, CLI tool, and test suites.

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Critical Requirements

**NEVER attempt to build this project without:**
- NVIDIA GPU with compute capability 6.0 or higher
- CUDA Toolkit 11.2+ installed and available in PATH
- Internet access for downloading dependencies via vcpkg
- At least 4GB free disk space for build artifacts

**Known compatibility issues that will cause build failures:**
- GCC 12+ (use GCC 9.x-11.x)
- Visual Studio 17.10+ (use 17.9 or earlier)  
- CUDA 12.5+ (use CUDA 12.4 or earlier)

## Getting Started

### Initial Setup
```bash
# Clone with recursive flag for vcpkg submodule - REQUIRED
git clone --recursive https://github.com/chrxh/alien.git
cd alien

# If already cloned without --recursive, initialize submodules
git submodule update --init --recursive
```

### System Dependencies (Ubuntu/Debian)
```bash
# Install build tools and OpenGL dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake
sudo apt-get install -y libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxext-dev libxfixes-dev libgl1-mesa-dev libglu1-mesa-dev

# Install CUDA Toolkit 11.2-12.4 (example for Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-compiler-12-4
```

### Bootstrap vcpkg Package Manager
```bash
cd external/vcpkg
./bootstrap-vcpkg.sh -disableMetrics
cd ../..
```

## Building the Project

### Configure and Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j8
```

**NEVER CANCEL: Build takes 45-90 minutes depending on system. ALWAYS set timeout to 120+ minutes.**

**Common build failures:**
- Missing CUDA: Install CUDA Toolkit and ensure nvcc is in PATH
- vcpkg download failures: Check internet connectivity and proxy settings
- Out of disk space: Ensure 4GB+ free space

### Build Outputs
After successful build, executables are located in `build/`:
- `alien` or `alien.exe`: Main GUI application
- `cli` or `cli.exe`: Command-line interface
- `EngineTests`: Unit tests for simulation engine
- `NetworkTests`: Unit tests for networking components

## Testing

### Run Unit Tests
```bash
cd build
./EngineTests
./NetworkTests
```
**NEVER CANCEL: Test suites take 15-30 minutes each. Set timeout to 45+ minutes.**

### CLI Testing
```bash
cd build
# Test CLI help
./cli --help

# Run sample simulation (requires .sim file)
./cli -i ../resources/autosave.sim -o test_output.sim -t 1000
```

## Running Applications

### GUI Application
```bash
cd build
./alien
```
**IMPORTANT: Must run from build directory - application needs access to resources/ folder.**

**GUI Requirements:**
- X11 display (for Linux)
- NVIDIA GPU with display capability
- OpenGL 3.3+ support

### Command-Line Interface
```bash
cd build
./cli -i input.sim -o output.sim -t [timesteps]
```

**CLI Parameters:**
- `-i`: Input simulation file (.sim)
- `-o`: Output simulation file (.sim) 
- `-t`: Number of timesteps to simulate

## Validation Scenarios

**ALWAYS test these scenarios after making changes:**

1. **Build Validation**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release -j8
   ```

2. **CLI Functionality Test**:
   ```bash
   cd build
   ./cli --help
   ./cli -i ../resources/autosave.sim -o test.sim -t 100
   ls -la test.sim test.sim.settings.json test.statistics.csv
   ```

3. **Unit Test Execution**:
   ```bash
   cd build  
   ./EngineTests --gtest_brief=1
   ./NetworkTests --gtest_brief=1
   ```

4. **GUI Startup Test** (if display available):
   ```bash
   cd build
   timeout 30s ./alien || echo "GUI started successfully"
   ```

## Development Workflow

### Code Organization
- `source/Base/`: Core utilities and base classes
- `source/EngineGpuKernels/`: CUDA kernels for simulation
- `source/EngineImpl/`: Simulation engine implementation  
- `source/EngineInterface/`: Engine API definitions
- `source/Gui/`: ImGui-based user interface
- `source/Cli/`: Command-line interface implementation
- `source/EngineTests/`: Unit tests for engine
- `source/NetworkTests/`: Unit tests for networking

### Making Changes
1. **Always build and test before making changes**
2. **Focus on specific modules** - avoid cross-cutting changes
3. **Run relevant tests** after modifications
4. **Test CLI functionality** if changing engine components
5. **Verify GPU code compilation** if modifying CUDA kernels

### Common Development Tasks

**Simulation Engine Changes:**
```bash
# Build only engine components
cd build
make EngineImpl EngineGpuKernels
./EngineTests
```

**CLI Changes:**
```bash
# Build and test CLI
cd build  
make cli
./cli --help
./cli -i ../resources/autosave.sim -o test.sim -t 100
```

**GUI Changes:**
```bash
# Build GUI application
cd build
make alien
./alien
```

## Build Time Expectations

- **Initial vcpkg setup**: 30-60 minutes (network dependent)
- **Full build from scratch**: 45-90 minutes
- **Incremental builds**: 2-15 minutes
- **Unit test execution**: 15-30 minutes per suite
- **Large repository size**: ~1.1GB with dependencies

## Troubleshooting

### Build Failures
1. **Check CUDA installation**: `nvcc --version`
2. **Verify vcpkg bootstrap**: `external/vcpkg/vcpkg version`
3. **Check GCC version**: `gcc --version` (should be 9.x-11.x)
4. **Ensure internet access** for vcpkg downloads
5. **Check disk space**: Need 4GB+ free

### Runtime Issues
1. **GPU not found**: Verify NVIDIA drivers and CUDA installation
2. **Resources not found**: Always run from build directory
3. **Display issues**: Ensure X11 forwarding or local display

### Network Restrictions
If behind firewall/proxy:
- vcpkg downloads may fail
- Consider pre-built dependencies or alternative build environment
- Document network requirements for team environments

## File Locations Reference

```
alien/
├── README.md              # Main documentation
├── CMakeLists.txt         # Root build configuration
├── vcpkg.json            # Package dependencies
├── external/             # Third-party dependencies
│   └── vcpkg/           # Package manager (Git submodule)
├── source/              # All source code (338 files)
│   ├── Base/           # Core utilities
│   ├── EngineGpuKernels/ # CUDA simulation kernels
│   ├── EngineImpl/     # Engine implementation
│   ├── EngineInterface/ # Engine API
│   ├── Gui/           # User interface
│   ├── Cli/           # Command-line tool
│   ├── EngineTests/   # Unit tests
│   └── NetworkTests/  # Network tests
├── resources/           # Runtime resources (shaders, icons, samples)
├── scripts/            # Utility scripts
└── build/              # Build output directory
```

## Key Sample Commands Output

### Repository Structure
```bash
$ ls -la
total 32
drwxr-xr-x 8 user user 4096 date .
drwxr-xr-x 3 user user 4096 date ..
-rw-r--r-- 1 user user 1234 date CMakeLists.txt
-rw-r--r-- 1 user user 5678 date README.md
drwxr-xr-x 8 user user 4096 date external
drwxr-xr-x 2 user user 4096 date resources
drwxr-xr-x 11 user user 4096 date source
-rw-r--r-- 1 user user  890 date vcpkg.json
```

### Source File Count
```bash
$ find source -name "*.h" -o -name "*.cpp" | wc -l
338
```

### CLI Help Output
```bash
$ ./cli --help
Command-line interface for ALIEN v4.12.2
Usage: ./cli [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -i TEXT                     Specifies the name of the input file for the simulation to run. The corresponding *.settings.json should also be available.
  -o TEXT                     Specifies the name of the output file for the simulation. The *.settings.json and *.statistics.csv file will also be saved.
  -t INT                      The number of time steps to be calculated.
```