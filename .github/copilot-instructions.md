# GitHub Copilot Instructions for ALIEN

## General Project Overview
ALIEN is an artificial life simulator implemented in C++20 with CUDA acceleration. It features a sophisticated multi-layered architecture with GPU-accelerated simulation, Dear ImGui-based GUI, and comprehensive testing infrastructure.

## Repository Structure
- `external/`: External libraries which are not included via vcpkg
- `resources/`: Resource files needed by ALIEN during runtime
- `scripts/`: PHP scripts for server to handle download, upload or info requests
- `source/`: Contains the C++ and CUDA sources
- `source/Base/`: General utilities and base classes, mostly domain-agnostic
- `source/Cli/`: Command line interface for ALIEN
- `source/EngineGpuKernels/`: CUDA kernels and GPU services for simulation execution
- `source/EngineImpl/`: Implements interfaces from `EngineInterface/` by invoking GPU services
- `source/EngineInterface/`: Interfaces for (asynchronous) simulation data transfer and operations
- `source/EngineTestData/`: Provides `TestDataFactory` for test projects
- `source/EngineTests/`: Integration tests for the engine using Google Test
- `source/Gui/`: GUI project based on Dear ImGui with custom widgets
- `source/Network/`: Communication functionalities via HTTPS requests
- `source/NetworkTests/`: Test project for the network module
- `source/PersisterImpl/`: Implements asynchronous loading, saving, uploading and downloading
- `source/PersisterInterface/`: Interfaces for asynchronous data transfer to/from storage or network

## Architecture Patterns

### 3-Layer Data Architecture
The project uses a strict 3-layer data transformation pattern:
1. **Description Layer** (`Descriptions.h`, `GenomeDescription.h`): CPU-side data for GUI and serialization
2. **Transfer Object Layer** (`ObjectTO.h`, `GenomeTO.h`): Intermediate format for GPU data transfer
3. **GPU Object Layer** (`Object.cuh`, `Genome.cuh`): GPU-optimized data structures for kernels

### Interface-Implementation Separation
- **Interfaces** in `EngineInterface/`: Define contracts and data structures
- **Implementations** in `EngineImpl/`: Delegate to GPU services
- **GPU Services** in `EngineGpuKernels/`: Execute CUDA kernels

## Coding Standards and Best Practices

### File Extensions and Languages
- `.h`: C++ headers
- `.cpp`: C++ implementation files
- `.cuh`: CUDA headers (device code)
- `.cu`: CUDA implementation files

### Code Style (clang-format enforced)
- **Column limit**: 160 characters
- **Indentation**: 4 spaces, no tabs
- **Braces**: Allman style (opening brace on new line for classes/functions)
- **Pointer alignment**: Left (`int* ptr` not `int *ptr`)
- **Include order**: Headers grouped and sorted (corresponding header, std library, external, project modules)

### Naming Conventions
- **Classes**: PascalCase (`SimulationFacade`, `CellDescription`)
- **Variables/Functions**: camelCase (`cellId`, `processEvents()`)
- **Constants**: PascalCase in `Const` namespace (`Const::WindowAlpha`)
- **Private members**: Underscore prefix (`_cellId`, `_simulationFacade`)
- **CUDA files**: Use `.cuh`/`.cu` extensions, may have `__device__` functions

### Essential Macros and Patterns

#### MEMBER Macro Pattern
```cpp
// GOOD: Use MEMBER macro for builder pattern in description classes
struct CellDescription {
    MEMBER(CellDescription, uint64_t, id, 0);
    MEMBER(CellDescription, RealVector2D, pos, {});
    MEMBER(CellDescription, float, energy, 100.0f);
};

// Usage:
auto cell = CellDescription().id(123).pos({10.0f, 20.0f}).energy(50.0f);

// BAD: Manual getter/setter implementation
struct CellDescription {
    uint64_t _id = 0;
    CellDescription& setId(uint64_t id) { _id = id; return *this; }
    uint64_t getId() const { return _id; }
};
```

#### Error Handling
```cpp
// GOOD: Use CHECK macro for assertions
CHECK(cellId != 0);
CHECK(energy >= 0.0f);

// GOOD: Use THROW_NOT_IMPLEMENTED for placeholder methods
void processComplexFeature() {
    THROW_NOT_IMPLEMENTED();
}

// BAD: Manual assertion or exception throwing
if (cellId == 0) {
    throw std::runtime_error("cell ID cannot be zero");
}
```

#### CUDA Device Code
```cpp
// GOOD: Proper CUDA atomics and thread safety
__device__ __inline__ bool tryLock() {
    auto result = 0 == atomicExch(&locked, 1);
    if (result) {
        __threadfence();
    }
    return result;
}

// GOOD: CUDA kernel function signature
__global__ void processSimulationKernel(SimulationData data, int2 rectUpperLeft, int2 rectLowerRight);

// BAD: Missing __device__ qualifier for device-only functions
inline bool tryLock() { return atomicExch(&locked, 1) == 0; } // Will fail in device code
```

## How to Use Copilot Effectively in ALIEN

### 1. Leverage Context-Aware Completion
- **DO**: Keep related files open when working on data transformations (e.g., `Descriptions.h`, `ObjectTO.h`, `Object.cuh`)
- **DO**: Include relevant header files in your prompt context
- **DON'T**: Work on GPU code without having the corresponding CPU interface visible

### 2. Architecture-Aware Suggestions
When Copilot suggests code, ensure it follows ALIEN's patterns:

```cpp
// GOOD: Copilot suggestion that follows ALIEN patterns
struct NewFeatureDescription {
    auto operator<=>(NewFeatureDescription const&) const = default;
    
    MEMBER(NewFeatureDescription, float, threshold, 0.5f);
    MEMBER(NewFeatureDescription, std::optional<int>, maxCount, std::nullopt);
};

// BAD: Copilot suggestion that ignores ALIEN patterns  
class NewFeature {
public:
    void setThreshold(float value) { threshold = value; }
    float getThreshold() const { return threshold; }
private:
    float threshold = 0.5f;
};
```

### 3. CUDA-Specific Guidance
- **Prompt**: "Create a CUDA kernel for [operation] following ALIEN's GPU object patterns"
- **Context**: Always include relevant `.cuh` files when working on GPU code
- **Validation**: Ensure `__device__` qualifiers and proper atomics usage

## Common Code Patterns and Examples

### Testing Patterns
```cpp
// GOOD: Follow existing test structure
class MyFeatureTests : public IntegrationTestFramework {
public:
    MyFeatureTests() : IntegrationTestFramework(getParameters()) {}
    
    static SimulationParameters getParameters() {
        SimulationParameters result;
        result.innerFriction = 0;
        // Configure test-specific parameters
        return result;
    }
};

TEST_F(MyFeatureTests, basicFunctionality) {
    // Arrange
    DataDescription data;
    data.addCell(CellDescription().setId(1).setPos({0, 0}));
    
    // Act
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    
    // Assert
    auto result = _simulationFacade->getSimulationData();
    EXPECT_EQ(1, result.cells.size());
}
```

### GUI Widget Patterns  
```cpp
// GOOD: Follow AlienGui patterns for custom widgets
if (AlienGui::SliderFloat(
    AlienGui::SliderFloatParameters()
        .name("Energy")
        .min(0.0f)
        .max(1000.0f)
        .textWidth(scale(100)),
    &energy)) {
    // Handle value change
}

if (AlienGui::Button("Process##unique_id")) {
    // Handle button click - note unique ID to avoid conflicts
}

// GOOD: Use proper scaling for consistent UI
auto width = scale(200.0f);
auto height = scale(100.0f);
ImGui::SetNextWindowSize({width, height});

// BAD: Using raw ImGui instead of AlienGui widgets
if (ImGui::SliderFloat("Energy", &energy, 0.0f, 1000.0f)) {
    // This ignores ALIEN's styling and scaling
}
```

### Singleton Patterns
```cpp
// GOOD: Singleton class using ALIEN's MAKE_SINGLETON macro
class MyService {
    MAKE_SINGLETON(MyService);

public:
    void processData(DataDescription const& data);
    
private:
    // Implementation (constructor automatically private via macro)
};

// GOOD: Singleton without default construction
class MyWindowService {
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MyWindowService);

public:
    void initialize(SimulationFacade simulationFacade);
    
private:
    MyWindowService(/* custom constructor parameters */);
    // Implementation
};

// Usage:
auto& service = MyService::get();
service.processData(data);
```

## What Copilot Should Suggest

### ✅ GOOD Suggestions
- Use of MEMBER macro for description classes
- Proper CUDA `__device__` qualifiers and atomics
- Integration test patterns matching existing tests
- AlienGui widget usage with proper scaling
- Three-layer data transformation patterns
- Use of MAKE_SINGLETON/MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION macros
- Proper include ordering following clang-format rules

### ❌ BAD Suggestions to Reject
- Manual getter/setter methods instead of MEMBER macro
- Raw pointers instead of smart pointers for CPU code
- Missing `__device__` qualifiers in CUDA code
- Standard widgets instead of AlienGui alternatives
- Breaking the interface-implementation pattern
- Ignoring the 3-layer data architecture
- Including platform-specific code without proper guards

## Configuration and Workflow

### Recommended Copilot Settings
- Enable "Enhanced completions for C++"
- Use workspace context for better architectural awareness
- Keep related architecture files open during development

### Code Review Checklist
- [ ] Does the suggestion follow ALIEN's 3-layer data pattern?
- [ ] Are CUDA functions properly marked with `__device__`?
- [ ] Does GUI code use AlienGui widgets with proper scaling?
- [ ] Are new description classes using MEMBER macros?
- [ ] Do tests follow the IntegrationTestFramework pattern?
- [ ] Is the include order correct per clang-format rules?
- [ ] Are atomics used properly in CUDA code?

### Building and Testing
```bash
# Build the project
cmake --build build --parallel

# Run tests
cd build && ctest --parallel

# Format code
clang-format -i source/**/*.{h,cpp}
```

## Reporting Issues and Improvements

### When Copilot Suggests Problematic Code
1. **Document the pattern**: Create an issue describing the problematic suggestion
2. **Provide context**: Include the surrounding code and expected pattern
3. **Share examples**: Show both the bad suggestion and correct ALIEN pattern
4. **Update instructions**: Contribute improvements to this file

### Contributing to These Instructions
- Add new patterns as the codebase evolves
- Include examples from actual code when possible
- Keep language clear and actionable
- Test suggestions against real development scenarios

## Quick Reference

### Key Files to Keep Open
- `source/EngineInterface/Descriptions.h` - Main data structures
- `source/EngineGpuKernels/Object.cuh` - GPU data structures  
- `source/Base/Macros.h` - Essential macros
- `source/_clang-format` - Code style configuration

### Common Prompts
- "Create a description class for [feature] using ALIEN patterns"
- "Implement a CUDA kernel for [operation] following ALIEN conventions"
- "Add a test for [feature] using IntegrationTestFramework"
- "Create a GUI widget for [control] using AlienGui patterns"


