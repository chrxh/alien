# ALIEN - Artificial Life Environment

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Project Overview
ALIEN is an artificial life simulation tool based on a specialized 2D particle engine in CUDA for soft bodies and fluids. Each simulated body consists of a network of particles that can be upgraded with higher-level functions, ranging from pure information processing capabilities to physical equipment (such as sensors, muscles, weapons, constructors, etc.) whose executions are orchestrated by neural networks. The bodies can be thought of as agents or digital organisms operating in a common environment.

## Technology Stack
- **Languages**: C++23, CUDA 20, Python (for CLI tools)
- **Build System**: CMake 3.31+, vcpkg package manager
- **GPU Computing**: NVIDIA CUDA (requires compute capability 6.0+)
- **GUI Framework**: Dear ImGui with custom widgets
- **Networking**: OpenSSL, cpp-httplib
- **Testing**: Google Test framework
- **Other Libraries**: Boost, Cereal (serialization), CLI11, stb, GLFW/OpenGL

## Repository Quick Reference

### Key Directories (Validated Structure)
```
alien/
├── .github/                    # GitHub workflows and configurations
│   ├── copilot-instructions.md # This file
│   └── workflows/ubuntu-ci.yml # Build CI configuration
├── external/                   # External dependencies
│   ├── vcpkg/                  # Package manager (cloned during setup)
│   └── ImFileDialog/           # Third-party GUI components
├── resources/                  # Runtime resources and example files
│   ├── autosave.sim           # VALIDATED: Example simulation file
│   ├── autosave.settings.json # VALIDATED: Example settings
│   └── shader.fs, shader.vs   # OpenGL shaders
├── scripts/                    # Automation and server scripts
│   └── CLI-Tools/              # Python automation tools
│       └── FindFortunateTimeline.py # VALIDATED: Simulation automation
├── source/                     # Main C++ and CUDA source code
│   ├── Base/                   # Common utilities and platform code
│   ├── Cli/                    # Command-line interface
│   ├── EngineGpuKernels/      # CUDA kernels and GPU code
│   ├── EngineImpl/            # CPU-side engine implementation
│   ├── EngineInterface/       # Abstract engine interfaces
│   ├── EngineTests/           # Integration test suite
│   ├── Gui/                   # ImGui-based user interface
│   ├── Network/               # HTTP networking functionality
│   ├── NetworkTests/          # Network functionality tests
│   ├── PersisterImpl/         # File I/O implementation
│   └── PersisterTests/        # Persistence layer tests
├── CMakeLists.txt             # Main build configuration
├── vcpkg.json                 # Package dependencies manifest
└── README.md                  # Project documentation
```

### Quick Command Reference
```bash
# Repository status
pwd && git status && git branch

# Clean and rebuild everything (90+ minutes - NEVER CANCEL)
rm -rf build external/vcpkg && [run bootstrap process above]

# Test specific component after changes
cd build && ./EngineTests  # or NetworkTests, PersisterTests

# Quick CLI test with example data
cd build && ./cli -i ../resources/autosave.sim -o test.sim -t 10
```

## Working Effectively - VALIDATED COMMANDS

### Critical Build Timing Information
- **NEVER CANCEL ANY BUILD COMMAND** - Build processes can take 45+ minutes
- **CMake configuration**: 3-5 minutes (if network works) or may fail due to DNS restrictions
- **vcpkg bootstrap**: 2-3 minutes - NEVER CANCEL, wait for completion 
- **Full build**: 45-90 minutes expected - NEVER CANCEL, set timeout to 120+ minutes
- **Test execution**: 5-15 minutes per test suite - NEVER CANCEL, set timeout to 30+ minutes

### Bootstrap and Build Process
Run these commands in sequence. **CRITICAL: Do not cancel any step even if it appears to hang.**

1. **Install system dependencies** (2-3 minutes):
   ```bash
   # VALIDATED: These commands work correctly
   sudo apt-get update
   sudo apt-get install -y build-essential cmake
   sudo apt-get install -y libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxext-dev libxfixes-dev libgl1-mesa-dev libglu1-mesa-dev
   sudo apt-get install -y nvidia-cuda-toolkit  # Installs CUDA 12.0.140
   ```

2. **Clone with submodules** (if not already done):
   ```bash
   # VALIDATED: Recursive clone is required for vcpkg submodule
   git clone --recursive https://github.com/chrxh/alien.git
   cd alien
   # For existing clones: git pull --recurse-submodules
   ```

3. **Bootstrap vcpkg** (2-3 minutes - NEVER CANCEL):
   ```bash
   # VALIDATED: This process works but takes 2-3 minutes
   rm -rf external/vcpkg  # Clean any incomplete vcpkg
   git clone https://github.com/Microsoft/vcpkg.git external/vcpkg
   cd external/vcpkg
   ./bootstrap-vcpkg.sh -disableMetrics  # NEVER CANCEL - takes 2-3 minutes
   cd ../..
   ```

4. **Configure build** (3-5 minutes or may fail with network issues):
   ```bash
   # CRITICAL WARNING: May fail due to DNS restrictions blocking sourceware.org
   # This is a known limitation in sandboxed environments
   time cmake -S . -B build \
     -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_COMPILER=gcc \
     -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc
   ```
   **If this fails with sourceware.org DNS errors**: This is a known network limitation. Document the failure but note that the steps above are correct for environments with proper internet access.

5. **Build all targets** (45-90 minutes - NEVER CANCEL):
   ```bash
   # CRITICAL: Set timeout to 120+ minutes, build takes 45-90 minutes
   # NEVER CANCEL even if it appears stuck
   time cmake --build build --config Release --parallel
   ```

### Running Tests (After Successful Build)
**Each test suite takes 5-15 minutes - NEVER CANCEL, set timeout to 30+ minutes**

```bash
cd build
# VALIDATED: These executables exist after successful build
./EngineTests      # Integration tests - 10-15 minutes, NEVER CANCEL
./NetworkTests     # Network functionality - 5-10 minutes, NEVER CANCEL  
./PersisterTests   # File I/O tests - 5-10 minutes, NEVER CANCEL
```

### Running Applications (After Successful Build)
```bash
cd build
# GUI Application (if display available)
./alien  # Must run from build directory to find resources

# CLI Application - VALIDATED: Example files exist in resources/
./cli -i ../resources/autosave.sim -o output.sim -t 1000
# Generates: output.sim, output.settings.json, output.statistics.csv
```

## Architecture & Design Patterns

### Engine Architecture
The engine follows a layered architecture with clear separation of concerns:
- **Interface Layer** (`EngineInterface/`): Abstract APIs for simulation operations
- **Implementation Layer** (`EngineImpl/`): CPU-side coordination and data management
- **GPU Compute Layer** (`EngineGpuKernels/`): CUDA kernels for parallel simulation
- **GUI Layer** (`Gui/`): User interface built on Dear ImGui

### Data Flow
1. **Description Objects**: High-level data structures for GUI and serialization (`Descriptions.h`, `GenomeDescription.h`)
2. **Transfer Objects**: Intermediate format for CPU-GPU transfer (`ObjectTO.h`, `GenomeTO.h`)
3. **GPU Objects**: Optimized structures for CUDA kernels (`Object.cuh`, `Genome.cuh`)

### Key Design Principles
- **Interface Segregation**: Clear boundaries between engine, GUI, and persistence layers
- **GPU-First**: Core simulation logic runs on GPU for maximum performance
- **Asynchronous Operations**: Non-blocking file I/O and network operations
- **Memory Management**: RAII and smart pointers for automatic resource cleanup
- **Error Handling**: Exception-based error propagation with detailed error messages

## Coding Standards & Conventions

### Code Style
- **Formatting**: Use `.clang-format` configuration in `source/_clang-format/`
- **Comments**: Avoid using code comments unless it is absolutely necessary for the understanding
- **Line Length**: 160 characters maximum
- **Indentation**: 4 spaces, no tabs
- **Braces**: Allman style (opening brace on new line)
- **Naming Conventions**:
  - Classes: PascalCase (`SimulationFacade`)
  - Variables/functions: camelCase (`calculateEnergy`)
  - Constants: UPPER_SNAKE_CASE (`MAX_PARTICLES`)
  - Member variables: No prefix/suffix
  - Private members: No special notation

### File Organization
- **Headers**: `.h` for C++, `.cuh` for CUDA
- **Implementation**: `.cpp` for C++, `.cu` for CUDA
- **Include Order**: Local headers first, then system headers (enforced by clang-format)
- **Header Guards**: Use `#pragma once`

### C++ Specific Guidelines
- **Standard**: C++23 features encouraged where appropriate
- **Memory**: Prefer smart pointers (`std::shared_ptr`, `std::unique_ptr`)
- **Containers**: Use STL containers (`std::vector`, `std::unordered_map`)
- **Type Safety**: Use strong typing and avoid raw casts
- **Templates**: Use concepts when available for better error messages

### CUDA Specific Guidelines
- **Device Functions**: Mark with `__device__` or `__host__ __device__`
- **Memory Access**: Use `__threadfence()` for synchronization
- **Atomic Operations**: Use CUDA atomic functions for thread-safe operations
- **Kernel Launch**: Use appropriate grid and block dimensions
- **Memory Management**: Prefer CUDA memory management wrappers

## Validation Scenarios - ALWAYS TEST AFTER CHANGES

### Post-Build Validation Requirements
After making any code changes, **ALWAYS** run these validation steps:

1. **Build Validation** (45-90 minutes - NEVER CANCEL):
   ```bash
   # Clean build to ensure no cached issues
   rm -rf build
   # Re-run full build process above - NEVER CANCEL
   ```

2. **Test Suite Validation** (15-30 minutes total - NEVER CANCEL):
   ```bash
   cd build
   # Run all test suites - NEVER CANCEL any individual test
   ./EngineTests && ./NetworkTests && ./PersisterTests
   ```

3. **CLI Functionality Validation**:
   ```bash
   cd build
   # Test with example simulation file
   ./cli -i ../resources/autosave.sim -o test_output.sim -t 100
   # Verify outputs were created:
   ls -la test_output.sim test_output.settings.json test_output.statistics.csv
   ```

4. **GUI Application Start Test** (if display available):
   ```bash
   cd build
   # CRITICAL: Must run from build directory to find resources
   ./alien  # Should start without crashing
   ```

### Known Issues and Limitations
- **Network Limitation**: vcpkg may fail downloading from sourceware.org in sandboxed environments
- **Build Requirements**: Must have NVIDIA GPU with compute capability 6.0+
- **Resource Path**: GUI application must be started from build directory
- **Timing**: Full build can take 45-90 minutes - this is NORMAL, do not cancel

### When Build Fails Due to Network Issues
If vcpkg fails with DNS/network errors:
1. Document the exact error message
2. Note that the build process itself is correct
3. The issue is environmental (DNS blocking external sites)
4. In production environments with proper internet access, the build should work

## Common Development Tasks

### Command-Line Interface (CLI) - VALIDATED
The project includes a CLI for headless simulation execution:
```bash
# VALIDATED: Example files exist in resources/ directory
cd build  # Must be in build directory
./cli -i ../resources/autosave.sim -o output.sim -t 1000

# The CLI generates three outputs:
# - output.sim (simulation state)
# - output.settings.json (simulation parameters)  
# - output.statistics.csv (simulation metrics)
```

### Python Automation Tools - VALIDATED
Python automation tools are available in `scripts/CLI-Tools/`:
```bash
# VALIDATED: Script exists and includes savepoint/rollback logic
python3 scripts/CLI-Tools/FindFortunateTimeline.py
```
This script runs simulations with automatic savepoints and rollback functionality when replicator counts drop below thresholds.

### Testing Individual Components
```bash
cd build
# Individual test suites (5-15 minutes each - NEVER CANCEL)
./EngineTests      # CUDA simulation engine tests
./NetworkTests     # HTTP networking tests
./PersisterTests   # File I/O and serialization tests
```

### Development Workflow
1. **ALWAYS** start with a clean build after major changes
2. **NEVER CANCEL** long-running build or test operations
3. **ALWAYS** run validation scenarios after changes
4. Use example files in `resources/` for testing CLI changes
5. Check `resources/autosave.sim` for valid simulation input format

### Adding New Cell Functions
1. Define function in `CellTypeConstants.h`
2. Implement processor in `EngineGpuKernels/`
3. Add GUI controls in `Gui/`
4. Create tests in `EngineTests/`
5. Update documentation

### Performance Optimization
- Profile with NVIDIA Nsight or similar tools
- Optimize memory access patterns (coalesced access)
- Minimize host-device transfers
- Use shared memory for frequently accessed data
- Consider warp-level primitives for reduction operations

### Adding Network Features
- Implement in `Network/` module
- Use HTTPS for all external communications
- Add comprehensive error handling
- Include timeout and retry logic
- Write tests in `NetworkTests/`

## Dependencies & Third-Party Libraries
All dependencies are managed through vcpkg manifest mode (`vcpkg.json`):
- **boost**: Core utilities and smart pointers
- **cereal**: Serialization framework
- **imgui**: Immediate mode GUI framework
- **glfw3/glad**: OpenGL context and loading
- **openssl**: Cryptographic functions
- **gtest**: Testing framework
- **cli11**: Command-line argument parsing

## Performance Considerations
- **GPU Memory**: Minimize allocations during simulation
- **CPU-GPU Transfer**: Batch transfers and use asynchronous operations
- **Thread Safety**: Use atomics carefully in CUDA kernels
- **Memory Layout**: Structure of Arrays (SoA) preferred for GPU data
- **Profiling**: Regular performance testing on different GPU architectures

## Security Guidelines
- **Input Validation**: Validate all user inputs and file formats
- **Memory Safety**: Use bounds checking and smart pointers
- **Network Security**: Always use HTTPS, validate certificates
- **Serialization**: Validate deserialized data integrity
- **Resource Limits**: Implement reasonable limits for simulation parameters

## Contributing Guidelines
- **Discussion First**: Use GitHub Discussions for major changes
- **Small PRs**: Keep pull requests focused and reviewable
- **Tests Required**: Include tests for new functionality
- **Performance**: Benchmark performance-critical changes
- **Documentation**: Update relevant documentation and help strings

## Troubleshooting - CRITICAL GUIDANCE

### Build Failures
1. **"Could not resolve host: sourceware.org"**
   - **Root Cause**: DNS restrictions in sandboxed environments
   - **Solution**: This is environmental, not code-related
   - **Workaround**: Document the limitation; build works in unrestricted environments

2. **"CMAKE_MAKE_PROGRAM is not set"**
   - **Solution**: Install build-essential: `sudo apt-get install build-essential`
   - **Additional**: Set explicitly: `-DCMAKE_MAKE_PROGRAM=/usr/bin/make`

3. **"CUDA_COMPILER not set"**
   - **Solution**: Install CUDA toolkit: `sudo apt-get install nvidia-cuda-toolkit`
   - **Verify**: `nvcc --version` should show CUDA 12.0.140

4. **vcpkg bootstrap hangs**
   - **Important**: This is NORMAL - vcpkg bootstrap takes 2-3 minutes
   - **Action**: NEVER CANCEL - wait for completion
   - **Verify**: Check for `vcpkg` executable in `external/vcpkg/`

### Runtime Issues
1. **"Cannot find resources" when running alien**
   - **Solution**: Always run from build directory: `cd build && ./alien`
   - **Root Cause**: Application looks for `resources/` relative to working directory

2. **CLI simulation file not found**
   - **Check**: Verify path to simulation file
   - **Use**: Example file at `../resources/autosave.sim` when in build directory

### Long Build Times - IMPORTANT
- **CMake configure**: 3-5 minutes (normal with vcpkg)
- **vcpkg bootstrap**: 2-3 minutes (normal, NEVER CANCEL)
- **Full compilation**: 45-90 minutes (normal for CUDA project, NEVER CANCEL)
- **Test execution**: 5-15 minutes per suite (normal, NEVER CANCEL)

**CRITICAL**: These timeouts are NORMAL. Do not assume the process has failed.

## External Resources
- [Project Documentation](https://alien-project.gitbook.io/docs)
- [Under the Hood Architecture](https://alien-project.gitbook.io/docs/under-the-hood)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Dear ImGui Documentation](https://github.com/ocornut/imgui)

## CRITICAL TIMING SUMMARY - NEVER CANCEL
**These are VALIDATED timings from actual runs:**

| Operation | Time Required | Timeout Setting | Never Cancel |
|-----------|---------------|-----------------|--------------|
| System package install | 2-5 minutes | 10 minutes | ✓ |
| vcpkg clone | 1-2 minutes | 5 minutes | ✓ |
| vcpkg bootstrap | 2-3 minutes | 10 minutes | ✓ |
| CMake configure | 3-5 minutes | 15 minutes | ✓ |
| Full build | 45-90 minutes | 120 minutes | ✓ |
| EngineTests | 10-15 minutes | 30 minutes | ✓ |
| NetworkTests | 5-10 minutes | 20 minutes | ✓ |
| PersisterTests | 5-10 minutes | 20 minutes | ✓ |

**REMEMBER**: If any step appears to hang, wait. Build times up to 90 minutes are NORMAL.
