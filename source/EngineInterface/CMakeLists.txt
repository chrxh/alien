
add_library(EngineInterface
    ArraySizesForGpuEntities.h
    ArraySizesForTOs.h
    CellTypeConstants.h
    CellTypeStrings.h
    Colors.h
    CudaSettings.h
    DataPointCollection.cpp
    DataPointCollection.h
    Definitions.h
    DescEditService.cpp
    DescEditService.h
    Desc.cpp
    Desc.h
    EngineConstants.h
    GenomeDesc.cpp
    GenomeDesc.h
    GenomeDescEditService.cpp
    GenomeDescEditService.h
    GenomeDescHash.h
    GenomeDescInfoService.cpp
    GenomeDescInfoService.h
    GenomeDescValidationService.cpp
    GenomeDescValidationService.h
    GeometryBuffers.cpp
    GeometryBuffers.h
    Ids.h
    InspectedEntityIds.h
    LocationHelper.cpp
    LocationHelper.h
    MutationType.h
    NeuralNetWeight.h
    NumberGenerator.cpp
    NumberGenerator.h
    ParametersEditService.cpp
    ParametersEditService.h
    ParametersFilter.h
    ParametersValidationService.cpp
    ParametersValidationService.h
    PreviewDesc.cpp
    PreviewDesc.h
    PreviewDescConverterService.cpp
    PreviewDescConverterService.h
    SelectionShallowData.h
    SettingsForSimulation.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.cpp
    ShapeGenerator.h
    SimulationFacade.cpp
    SimulationFacade.h
    SimulationParameters.cpp
    SimulationParameters.h
    SimulationParametersTypes.h
    SimulationParametersUpdateConfig.h
    SimulationParametersSpecification.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    SpecificationEvaluationService.cpp
    SpecificationEvaluationService.h
    SpecificationFilterService.cpp
    SpecificationFilterService.h
    StatisticsConverterService.cpp
    StatisticsConverterService.h
    StatisticsHistory.cpp
    StatisticsHistory.h
    StatisticsRawData.h)

target_link_libraries(EngineInterface Base)

target_link_libraries(EngineInterface Boost::boost)
target_link_libraries(EngineInterface glad::glad)

if (MSVC)
    target_compile_options(EngineInterface PRIVATE "/MP")
endif()
