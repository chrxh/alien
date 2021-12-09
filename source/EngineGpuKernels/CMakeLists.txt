
add_library(alien_engine_gpu_kernels_lib
    AccessKernels.cuh
    AccessTOs.cuh
    ActionKernels.cuh
    Array.cuh
    Base.cuh
    CellComputerFunction.cuh
    CellConnectionProcessor.cuh
    Cell.cuh
    CellFunctionData.cuh
    CellProcessor.cuh
    CleanupKernels.cuh
    CommunicatorFunction.cuh
    ConstantMemory.cuh
    ConstructorFunction.cuh
    CudaMemoryManager.cuh
    CudaMonitorData.cuh
    CudaSimulation.cu
    CudaSimulation.cuh
    DEBUG_cluster.cuh
    DebugKernels.cuh
    Definitions.cuh
    Definitions.h
    DllExport.h
    DynamicMemory.cuh
    EnergyGuidance.cuh
    Entities.cuh
    EntityFactory.cuh
    FlowFieldKernel.cuh
    HashMap.cuh
    HashSet.cuh
    List.cuh
    Macros.cuh
    Map.cuh
    MapSectionCollector.cuh
    Math.cuh
    MonitorKernels.cuh
    MuscleFunction.cuh
    Operation.cuh
    Particle.cuh
    ParticleProcessor.cuh
    Physics.cuh
    PropulsionFunction.cuh
    QuantityConverter.cuh
    RenderingData.cuh
    RenderingKernels.cuh
    ScannerFunction.cuh
    SelectionResult.cuh
    SensorFunction.cuh
    SimulationData.cuh
    SimulationKernels.cuh
    SimulationResult.cuh
    SpotCalculator.cuh
    Swap.cuh
    Textures.cuh
    Token.cuh
    TokenProcessor.cuh
    WeaponFunction.cuh)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/17520
set_property(TARGET alien_engine_gpu_kernels_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_path(STB_INCLUDE_DIRS "stb.h")
target_include_directories(alien_engine_gpu_kernels_lib PRIVATE ${STB_INCLUDE_DIRS})
