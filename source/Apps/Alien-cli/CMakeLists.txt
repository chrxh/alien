target_sources(alien-cli
PUBLIC
    CommandLineArguments.h
    CommandLineParser.cpp
    CommandLineParser.h
    ConsoleOutput.cpp
    ConsoleOutput.h
    LoginSession.cpp
    LoginSession.h
    Main.cpp
    PeriodicUploader.cpp
    PeriodicUploader.h
    SimulationRunner.cpp
    SimulationRunner.h)

target_link_libraries(alien-cli Base)
target_link_libraries(alien-cli ConsoleUi)
target_link_libraries(alien-cli EngineKernels)
target_link_libraries(alien-cli EngineImpl)
target_link_libraries(alien-cli Data)
target_link_libraries(alien-cli EngineInterface)
target_link_libraries(alien-cli Network)
target_link_libraries(alien-cli PersisterInterface)
target_link_libraries(alien-cli PersisterImpl)

target_include_directories(alien-cli
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(alien-cli cudart)
target_link_libraries(alien-cli Boost::boost)
target_link_libraries(alien-cli GTest::GTest GTest::Main)
target_link_libraries(alien-cli CLI11::CLI11)

target_precompile_headers(alien-cli PRIVATE
    <memory>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(alien-cli PRIVATE "/MP")
endif()
