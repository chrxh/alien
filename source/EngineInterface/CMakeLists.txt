
add_library(alien_engine_interface_lib
    ChangeDescriptions.cpp
    ChangeDescriptions.h
    Colors.h
    Definitions.h
    DescriptionHelper.cpp
    DescriptionHelper.h
    Descriptions.cpp
    Descriptions.h
    DllExport.h
    ElementaryTypes.h
    #EngineInterfaceSettings.cpp
    #EngineInterfaceSettings.h
    FlowFieldSettings.h
    GeneralSettings.h
    GpuSettings.h
    Metadata.h
    OverallStatistics.h
    Parser.cpp
    Parser.h
    SelectionShallowData.h
    Serializer.cpp
    Serializer.h
    Settings.h
    SimulationParameters.h
    SimulationParametersSpots.h
    SimulationParametersSpotValues.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    SymbolMap.h
    ZoomLevels.h)

target_link_libraries(alien_engine_interface_lib Boost::boost)
