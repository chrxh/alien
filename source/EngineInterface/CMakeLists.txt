
add_library(alien_engine_interface_lib
    Colors.h
    Definitions.h
    DescriptionHelper.cpp
    DescriptionHelper.h
    Descriptions.cpp
    Descriptions.h
    Enums.h
    FlowFieldSettings.h
    GeneralSettings.h
    GpuSettings.h
    InspectedEntityIds.h
    MonitorData.h
    OverlayDescriptions.h
    SelectionShallowData.h
    Serializer.cpp
    Serializer.h
    Settings.h
    SettingsParser.cpp
    SettingsParser.h
    ShallowUpdateSelectionData.h
    SimulationController.h
    SimulationParameters.h
    SimulationParametersSpots.h
    SimulationParametersSpotValues.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    ZoomLevels.h)

target_link_libraries(alien_engine_interface_lib Boost::boost)
target_link_libraries(alien_engine_interface_lib cereal)
target_link_libraries(alien ZLIB::ZLIB)

find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_include_directories(alien_engine_interface_lib PRIVATE ${ZSTR_INCLUDE_DIRS})
