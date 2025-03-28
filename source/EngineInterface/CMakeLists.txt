
add_library(EngineInterface
    ArraySizes.h
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
    ExpertSettingsToggles.h
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
    SimulationParametersTypes.h
    SimulationParametersUpdateConfig.h
    SimulationParametersValidationService.cpp
    SimulationParametersValidationService.h
    SimulationParametersZone.h
    SimulationParametersZoneEnabledValues.h
    SimulationParametersZoneValues.h
    SimulationParametersSpecification.h
    SimulationParametersSpecificationService.cpp
    SimulationParametersSpecificationService.h
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
