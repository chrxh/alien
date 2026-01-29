
add_library(EngineGpuKernels
    ActiveRadiationSources.cuh
    Array.cuh
    AttackerProcessor.cuh
    Base.cuh
    CellProcessor.cuh
    ClusterProcessor.cuh
    CommunicatorProcessor.cuh
    ConstantMemory.cu
    ConstantMemory.cuh
    Constants.cuh
    ConstructorHelper.cuh
    ConstructorProcessor.cuh
    CudaGeometryBuffers.cu
    CudaGeometryBuffers.cuh
    CudaMemoryManager.cuh
    CudaNumberGenerator.cuh
    CudaShapeGenerator.cuh
    CudaTOProvider.cu
    CudaTOProvider.cuh
    DataAccessKernels.cu
    DataAccessKernels.cuh
    DebugKernels.cu
    DebugKernels.cuh
    Definitions.cuh
    Definitions.h
    DensityMap.cuh
    DepotProcessor.cuh
    DetonatorProcessor.cuh
    DigestorProcessor.cuh
    EditKernels.cu
    EditKernels.cuh
    EnergyProcessor.cuh
    Entities.cu
    Entities.cuh
    EntityFactory.cuh
    ForceFieldKernels.cu
    ForceFieldKernels.cuh
    GarbageCollectorKernels.cu
    GarbageCollectorKernels.cuh
    GeneratorProcessor.cuh
    Genome.cuh
    GenomeTO.cuh
    GeometryKernels.cu
    GeometryKernels.cuh
    HashMap.cuh
    HashSet.cuh
    InjectorProcessor.cuh
    List.cuh
    Macros.cuh
    Map.cuh
    MapSectionCollector.cuh
    Math.cuh
    MaxAgeBalancer.cu
    MaxAgeBalancer.cuh
    MemoryProcessor.cuh
    MuscleProcessor.cuh
    MutationProcessor.cuh
    NeuronProcessor.cuh
    ObjectConnectionProcessor.cuh
    ObjectProcessor.cuh
    Operations.cuh
    ParameterCalculator.cuh
    Physics.cuh
    PreprocessedSimulationData.cuh
    ReconnectorProcessor.cuh
    SelectionKernels.cu
    SelectionKernels.cuh
    SelectionResult.cuh
    SensorProcessor.cuh
    SignalProcessor.cuh
    SimulationData.cu
    SimulationData.cuh
    SimulationKernels.cu
    SimulationKernels.cuh
    SimulationStatistics.cuh
    StatisticsKernels.cu
    StatisticsKernels.cuh
    TestKernels.cu
    TestKernels.cuh
    TOs.cuh
    TOProvider.cu
    TOProvider.cuh
    Util.cuh
    )

target_link_libraries(EngineGpuKernels Base)
target_link_libraries(EngineGpuKernels EngineInterface)
target_link_libraries(EngineGpuKernels OpenGL::GL)
target_link_libraries(EngineGpuKernels glad::glad)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/17520
set_property(TARGET EngineGpuKernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)