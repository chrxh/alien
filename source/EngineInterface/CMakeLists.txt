project(EngineInterface)

set(EngineInterface_SOURCES
    CellComputerCompiler.h
    CellComputerCompilerImpl.cpp
    CellComputerCompilerImpl.h
    ChangeDescriptions.cpp
    ChangeDescriptions.h
    Colors.h
    CompilerHelper.h
    Definitions.h
    DescriptionFactory.h
    DescriptionFactoryImpl.cpp
    DescriptionFactoryImpl.h
    DescriptionHelper.cpp
    DescriptionHelper.h
    DescriptionHelperImpl.cpp
    DescriptionHelperImpl.h
    Descriptions.cpp
    Descriptions.h
    DllExport.h
    ElementaryTypes.h
    EngineInterfaceBuilderFacade.h
    EngineInterfaceBuilderFacadeImpl.cpp
    EngineInterfaceBuilderFacadeImpl.h
    EngineInterfaceServices.cpp
    EngineInterfaceServices.h
    EngineInterfaceSettings.cpp
    EngineInterfaceSettings.h
    ExecutionParameters.h
    Metadata.h
    MonitorData.h
    PhysicalActions.h
    Physics.cpp
    Physics.h
    QuantityConverter.cpp
    QuantityConverter.h
    SerializationHelper.h
    Serializer.h
    SerializerImpl.cpp
    SerializerImpl.h
    SimulationAccess.h
    SimulationChanger.h
    SimulationChangerImpl.cpp
    SimulationChangerImpl.h
    SimulationContext.h
    SimulationController.h
    SimulationMonitor.h
    SimulationParametersCalculator.cpp
    SimulationParametersCalculator.h
    SimulationParameters.h
    SimulationParametersParser.cpp
    SimulationParametersParser.h
    SpaceProperties.cpp
    SpaceProperties.h
    SymbolTable.cpp
    SymbolTable.h
    ZoomLevels.h
)

add_library(EngineInterface SHARED ${EngineInterface_SOURCES})
add_library(ALiEn::EngineInterface ALIAS EngineInterface)

target_compile_definitions(EngineInterface PRIVATE ENGINEINTERFACE_LIB)

target_link_libraries(EngineInterface PUBLIC Qt6::Widgets Boost::serialization PRIVATE ALiEn::Base)

target_include_directories(EngineInterface
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
