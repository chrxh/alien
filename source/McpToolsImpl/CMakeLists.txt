add_library(McpToolsImpl
    McpCreatorTools.cpp
    McpCreatorTools.h
    McpEditTools.cpp
    McpEditTools.h
    McpGenomeTools.cpp
    McpGenomeTools.h
    McpInspectionTools.cpp
    McpInspectionTools.h
    McpMultiplierTools.cpp
    McpMultiplierTools.h
    McpNetworkTools.cpp
    McpNetworkTools.h
    McpParameterTools.cpp
    McpParameterTools.h
    McpPersisterTask.cpp
    McpPersisterTask.h
    McpSelectionTools.cpp
    McpSelectionTools.h
    McpSimulationTools.cpp
    McpSimulationTools.h
    McpTemporalTools.cpp
    McpTemporalTools.h
    McpToolsFacadeImpl.cpp
    McpToolsFacadeImpl.h
    McpViewTools.cpp
    McpViewTools.h)

target_link_libraries(McpToolsImpl Base)
target_link_libraries(McpToolsImpl Data)
target_link_libraries(McpToolsImpl EngineInterface)
target_link_libraries(McpToolsImpl McpToolsInterface)
target_link_libraries(McpToolsImpl Network)
target_link_libraries(McpToolsImpl PersisterInterface)
target_link_libraries(McpToolsImpl Boost::boost)
target_link_libraries(McpToolsImpl Boost::json)

target_precompile_headers(McpToolsImpl PRIVATE
    <memory>
    <optional>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(McpToolsImpl PRIVATE "/MP")
endif()
