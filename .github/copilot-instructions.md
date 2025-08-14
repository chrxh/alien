# ALIEN - Artificial Life Environment

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

## Repository Structure
- `external/`: External libraries not managed by vcpkg (including vcpkg submodule)
- `resources/`: Runtime resource files (shaders, fonts, etc.)
- `scripts/`: Server-side PHP scripts and CLI Python tools for automation
- `source/`: Main C++ and CUDA source code
  - `source/Base/`: Common utilities, math, logging, and platform-independent code
  - `source/Cli/`: Command-line interface for headless simulation execution
  - `source/EngineGpuKernels/`: CUDA kernels and GPU computation services
  - `source/EngineImpl/`: CPU-side implementation of engine interfaces
  - `source/EngineInterface/`: Abstract interfaces for simulation operations
  - `source/EngineTestData/`: Test data factory for integration tests
  - `source/EngineTests/`: Comprehensive integration test suite
  - `source/Gui/`: ImGui-based graphical user interface
  - `source/Network/`: HTTP client for cloud features and data sharing
  - `source/NetworkTests/`: Network functionality tests
  - `source/PersisterImpl/`: File I/O and serialization implementations
  - `source/PersisterInterface/`: Interfaces for data persistence
  - `source/PersisterTests/`: Persistence layer tests

## Development Environment Setup
1. **Prerequisites**: 
   - NVIDIA GPU with compute capability 6.0+ (GeForce 10 series or newer)
   - Latest NVIDIA graphics drivers
   - CMake 3.31+
   - CUDA Toolkit (version specified in CI)

2. **System Dependencies (Ubuntu/Debian)**
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

3. **Getting Sources**:
   ```bash
   git clone --recursive https://github.com/chrxh/alien.git
   cd alien
   ```
   Note: The `--recursive` parameter is necessary to check out the vcpkg submodule. Submodules are not updated by standard `git pull` - use `git pull --recurse-submodules`.

4. **Building**:
   ```bash
   # Bootstrap vcpkg (first time only, if not using CI approach)
   git clone https://github.com/Microsoft/vcpkg.git external/vcpkg
   cd external/vcpkg
   ./bootstrap-vcpkg.sh  # Linux/Mac
   # or .\bootstrap-vcpkg.bat  # Windows
   cd ../..
   
   # Configure and build
   cmake -S . -B build \
     -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ```

5. **Running Tests**:
   ```bash
   cd build
   # Run specific test executables
   ./EngineTests          # Integration tests for the simulation engine
   ./NetworkTests         # Network functionality tests  
   ./PersisterTests       # File I/O and persistence tests
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

## Testing Strategy

### Test Categories
1. **Unit Tests**: Individual component testing (preferred for new code)
2. **Integration Tests**: Cross-component interaction testing
3. **Performance Tests**: CUDA kernel and simulation performance validation
4. **Network Tests**: HTTP client and server communication testing

### Test Naming
- Test files: `*Tests.cpp`
- Test classes: `TestClassName`
- Test methods: `descriptive_test_name`

### Common Test Patterns
```cpp
// Integration test example
TEST_F(SimulationTest, cell_should_divide_when_energy_sufficient)
{
    // Arrange
    auto simulation = createSimulation();
    auto cell = createCellWithEnergy(100.0f);
    
    // Act
    simulation.calcTimestep();
    
    // Assert
    EXPECT_EQ(2, simulation.getNumCells());
}
```

## Common Development Tasks

### Command-Line Interface (CLI)
The project includes a CLI for headless simulation execution:
```bash
# Basic simulation run
./cli -i example.sim -o output.sim -t 1000

# The CLI generates three outputs:
# - output.sim (simulation state)
# - output.settings.json (simulation parameters)  
# - output.statistics.csv (simulation metrics)
```

### Automation Scripts
Python automation tools are available in `scripts/CLI-Tools/`:
- `FindFortunateTimeline.py`: Automated simulation with savepoints and rollback logic

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

## Debugging Tips
- **CUDA Debugging**: Use `cuda-gdb` or Nsight Compute for kernel debugging
- **Memory Leaks**: Run with CUDA memory checker
- **GUI Issues**: Check ImGui demo window for reference implementations
- **Network Problems**: Enable detailed logging in Network module
- **Build Issues**: Clean vcpkg cache and rebuild dependencies

## External Resources
- [Project Documentation](https://alien-project.gitbook.io/docs)
- [Under the Hood Architecture](https://alien-project.gitbook.io/docs/under-the-hood)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Dear ImGui Documentation](https://github.com/ocornut/imgui)


