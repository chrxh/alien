
add_library(ConsoleUi
    ConsoleLiveOutput.cpp
    ConsoleLiveOutput.h
    ConsoleSimulationPanel.cpp
    ConsoleSimulationPanel.h
    ConsoleWidgets.cpp
    ConsoleWidgets.h)

target_link_libraries(ConsoleUi Base)

target_precompile_headers(ConsoleUi PRIVATE
    <optional>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(ConsoleUi PRIVATE "/MP")
endif()
