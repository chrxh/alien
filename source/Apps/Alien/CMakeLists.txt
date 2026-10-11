target_sources(alien
PUBLIC
    HelpStrings.h
    Main.cpp)

target_link_libraries(alien Base)
target_link_libraries(alien EngineKernels)
target_link_libraries(alien EngineImpl)
target_link_libraries(alien GuiInterface)
target_link_libraries(alien GuiImpl)
target_link_libraries(alien McpToolsImpl)
target_link_libraries(alien PersisterImpl)
target_link_libraries(alien RenderingImpl)

target_include_directories(alien
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(alien cudart)

if (MSVC)
    target_compile_options(alien PRIVATE "/MP")
endif()

# Embed the application icon. The path is baked into the generated resource script because
# the resource compiler resolves relative paths against the build directory, not the source one.
if (WIN32)
    set(ALIEN_ICON_FILE "${CMAKE_SOURCE_DIR}/resources/images/alien.ico")
    configure_file(Resources.rc.in ${CMAKE_CURRENT_BINARY_DIR}/Resources.rc @ONLY)
    target_sources(alien PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Resources.rc)
endif()
