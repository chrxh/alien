
add_library(EngineInterface
    ArraySizesForGpu.h
    ArraySizesForTO.h
    CellTypeConstants.h
    CellTypeStrings.h
    Colors.h
    DataPointCollection.cpp
    DataPointCollection.h
    Definitions.h
    DescriptionEditService.cpp
    DescriptionEditService.h
    Descriptions.cpp
    Descriptions.h
    EngineConstants.h
    GenomeConstants.h
    GenomeDescriptionConverterService.cpp
    GenomeDescriptionConverterService.h
    GenomeDescriptions.h
    GpuSettings.h
    InspectedEntityIds.h
    LocationHelper.cpp
    LocationHelper.h
    MutationType.h
    OverlayDescriptions.h
    ParametersEditService.cpp
    ParametersEditService.h
    ParametersValidationService.cpp
    ParametersValidationService.h
    PreviewDescriptionService.cpp
    PreviewDescriptionService.h
    PreviewDescriptions.h
    SelectionShallowData.h
    SettingsForSimulation.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.cpp
    ShapeGenerator.h
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
    StatisticsConverterService.cpp
    StatisticsConverterService.h
    StatisticsHistory.cpp
    StatisticsHistory.h
    StatisticsRawData.h
    ZoomLevels.h)

target_link_libraries(EngineInterface Base)

target_link_libraries(EngineInterface Boost::boost)

if (MSVC)
    target_compile_options(EngineInterface PRIVATE "/MP")
endif()
