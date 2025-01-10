
add_library(EngineInterface
    ArraySizes.h
    CellFunctionConstants.h
    Colors.h
    DataPointCollection.cpp
    DataPointCollection.h
    Definitions.h
    DescriptionEditService.cpp
    DescriptionEditService.h
    Descriptions.cpp
    Descriptions.h
    EngineConstants.h
    Features.h
    GenomeConstants.h
    GenomeDescriptionService.cpp
    GenomeDescriptionService.h
    GenomeDescriptions.h
    GpuSettings.h
    InspectedEntityIds.h
    Motion.h
    MutationType.h
    OverlayDescriptions.h
    PreviewDescriptionService.cpp
    PreviewDescriptionService.h
    PreviewDescriptions.h
    RadiationSource.h
    SelectionShallowData.h
    SettingsForSimulation.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.cpp
    ShapeGenerator.h
    SimulationFacade.h
    SimulationParameters.h
    SimulationParametersEditService.cpp
    SimulationParametersEditService.h
    SimulationParametersZone.h
    SimulationParametersZoneActivatedValues.h
    SimulationParametersZoneValues.h
    SimulationParametersTypes.h
    SimulationParametersUpdateConfig.h
    SimulationParametersValidationService.cpp
    SimulationParametersValidationService.h
    SpaceCalculator.cpp
    SpaceCalculator.h
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
