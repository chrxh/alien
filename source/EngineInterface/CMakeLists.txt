
add_library(EngineInterface
    ArraySizes.h
    AuxiliaryData.h
    AuxiliaryDataParserService.cpp
    AuxiliaryDataParserService.h
    CellFunctionConstants.h
    Colors.h
    DataPointCollection.cpp
    DataPointCollection.h
    Definitions.h
    DescriptionEditService.cpp
    DescriptionEditService.h
    Descriptions.cpp
    Descriptions.h
    DeserializedSimulation.h
    EngineConstants.h
    Features.cpp
    Features.h
    GenomeConstants.h
    GenomeDescriptionService.cpp
    GenomeDescriptionService.h
    GenomeDescriptions.h
    GeneralSettings.h
    GpuSettings.h
    InspectedEntityIds.h
    LegacyAuxiliaryDataParserService.cpp
    LegacyAuxiliaryDataParserService.h
    Motion.h
    MutationType.h
    OverlayDescriptions.h
    PreviewDescriptionService.cpp
    PreviewDescriptionService.h
    PreviewDescriptions.h
    PropertyParser.h
    RadiationSource.h
    RawStatisticsData.h
    SelectionShallowData.h
    SerializerService.cpp
    SerializerService.h
    Settings.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.cpp
    ShapeGenerator.h
    SerializedSimulation.h
    SimulationController.h
    SimulationParameters.cpp
    SimulationParameters.h
    SimulationParametersSpot.h
    SimulationParametersSpotActivatedValues.h
    SimulationParametersSpotValues.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    StatisticsConverterService.cpp
    StatisticsConverterService.h
    StatisticsHistory.cpp
    StatisticsHistory.h
    ZoomLevels.h)

target_link_libraries(EngineInterface Boost::boost)
target_link_libraries(EngineInterface cereal)
target_link_libraries(EngineInterface ZLIB::ZLIB)

find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_include_directories(EngineInterface PRIVATE ${ZSTR_INCLUDE_DIRS})

if (MSVC)
    target_compile_options(EngineInterface PRIVATE "/MP")
endif()
