
add_library(alien_engine_interface_lib
    ArraySizes.h
    AuxiliaryData.h
    AuxiliaryDataParser.cpp
    AuxiliaryDataParser.h
    CellFunctionConstants.h
    Colors.h
    Definitions.h
    DescriptionHelper.cpp
    DescriptionHelper.h
    Descriptions.cpp
    Descriptions.h
    FundamentalConstants.h
    GenomeConstants.h
    GenomeDescriptionConverter.cpp
    GenomeDescriptionConverter.h
    GenomeDescriptions.h
    GeneralSettings.h
    GpuSettings.h
    InspectedEntityIds.h
    MutationType.h
    OverlayDescriptions.h
    PreviewDescriptionConverter.cpp
    PreviewDescriptionConverter.h
    PreviewDescriptions.h
    RadiationSource.h
    SelectionShallowData.h
    Serializer.cpp
    Serializer.h
    Settings.h
    ShallowUpdateSelectionData.h
    SimulationController.h
    SimulationParameters.h
    SimulationParametersSpot.h
    SimulationParametersSpotActivatedValues.h
    SimulationParametersSpotValues.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    StatisticsData.h
    ZoomLevels.h)

target_link_libraries(alien_engine_interface_lib Boost::boost)
target_link_libraries(alien_engine_interface_lib cereal)
target_link_libraries(alien ZLIB::ZLIB)

find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_include_directories(alien_engine_interface_lib PRIVATE ${ZSTR_INCLUDE_DIRS})

if (MSVC)
    target_compile_options(alien_engine_interface_lib PRIVATE "/MP")
endif()
