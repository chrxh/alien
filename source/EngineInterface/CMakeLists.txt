
add_library(alien_engine_interface_lib
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
    EngineConstants.h
    ExportService.cpp
    ExportService.h
    GenomeConstants.h
    GenomeDescriptionService.cpp
    GenomeDescriptionService.h
    GenomeDescriptions.h
    GeneralSettings.h
    GpuSettings.h
    InspectedEntityIds.h
    Motion.h
    MutationType.h
    OverlayDescriptions.h
    PreviewDescriptionService.cpp
    PreviewDescriptionService.h
    PreviewDescriptions.h
    RadiationSource.h
    RawStatisticsData.h
    SelectionShallowData.h
    SerializerService.cpp
    SerializerService.h
    Settings.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.cpp
    ShapeGenerator.h
    SimulationController.h
    SimulationParameters.h
    SimulationParametersSpot.h
    SimulationParametersSpotActivatedValues.h
    SimulationParametersSpotValues.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    StatisticsHistory.cpp
    StatisticsHistory.h
    ZoomLevels.h)

target_link_libraries(alien_engine_interface_lib Boost::boost)
target_link_libraries(alien_engine_interface_lib cereal)
target_link_libraries(alien ZLIB::ZLIB)

find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_include_directories(alien_engine_interface_lib PRIVATE ${ZSTR_INCLUDE_DIRS})

if (MSVC)
    target_compile_options(alien_engine_interface_lib PRIVATE "/MP")
endif()
