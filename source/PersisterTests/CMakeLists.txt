target_sources(PersisterTests
PUBLIC
    SerializerServiceTest.cpp
    Testsuite.cpp)

target_link_libraries(PersisterTests Base)
target_link_libraries(PersisterTests EngineGpuKernels)
target_link_libraries(PersisterTests EngineImpl)
target_link_libraries(PersisterTests EngineInterface)
target_link_libraries(PersisterTests EngineTestData)
target_link_libraries(PersisterTests PersisterInterface)

target_include_directories(PersisterTests
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(PersisterTests cudart)
target_link_libraries(PersisterTests Boost::boost)
target_link_libraries(PersisterTests OpenGL::GL OpenGL::GLU)
target_link_libraries(PersisterTests GLEW::GLEW)
target_link_libraries(PersisterTests glfw)
target_link_libraries(PersisterTests glad::glad)
target_link_libraries(PersisterTests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(PersisterTests PRIVATE "/MP")
endif()
