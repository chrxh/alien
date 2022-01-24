
add_library(alien_engine_interface_lib
    ShallowUpdateSelectionData.h
    CellComputationCompiler.cpp
    CellComputationCompiler.h
    CellInstruction.h
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
    Metadata.h
    OverallStatistics.h
    OverlayDescriptions.h
    Parser.cpp
    Parser.h
    SelectionShallowData.h
    Serializer.cpp
    Serializer.h
    Settings.h
    SimulationController.h
    SimulationParameters.h
    SimulationParametersSpots.h
    SimulationParametersSpotValues.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    SymbolMap.cpp
    SymbolMap.h
    ZoomLevels.h)

target_link_libraries(alien_engine_interface_lib Boost::boost)
target_link_libraries(alien_engine_interface_lib cereal)
