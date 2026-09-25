add_library(McpTools
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
    McpToolContext.h
    McpToolsService.cpp
    McpToolsService.h)

target_link_libraries(McpTools Base)
target_link_libraries(McpTools EngineInterface)
target_link_libraries(McpTools Network)
target_link_libraries(McpTools PersisterInterface)
target_link_libraries(McpTools Boost::boost)
target_link_libraries(McpTools Boost::json)

target_precompile_headers(McpTools PRIVATE
    <memory>
    <optional>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(McpTools PRIVATE "/MP")
endif()
