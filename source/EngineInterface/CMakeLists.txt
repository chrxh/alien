
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
    DescriptionEditService.cpp
    DescriptionEditService.h
    Description.cpp
    Description.h
    EngineConstants.h
    GenomeDescription.cpp
    GenomeDescription.h
    GenomeDescriptionEditService.cpp
    GenomeDescriptionEditService.h
    GenomeDescriptionHash.h
    GenomeDescriptionInfoService.cpp
    GenomeDescriptionInfoService.h
    GenomeDescriptionValidationService.cpp
    GenomeDescriptionValidationService.h
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
    PreviewDescription.cpp
    PreviewDescription.h
    PreviewDescriptionConverterService.cpp
    PreviewDescriptionConverterService.h
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
