add_library(McpToolsImpl
    McpCreatorTools.cpp
    McpCreatorTools.h
    McpMultiplierTools.cpp
    McpMultiplierTools.h
    McpParameterTools.cpp
    McpParameterTools.h
    McpSelectionTools.cpp
    McpSelectionTools.h
    McpSimulationTools.cpp
    McpSimulationTools.h
    McpToolsFacadeImpl.cpp
    McpToolsFacadeImpl.h)

target_link_libraries(McpToolsImpl Base)
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
