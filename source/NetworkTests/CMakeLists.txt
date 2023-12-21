target_sources(NetworkTests
PUBLIC
    BrowserDataServiceTests.cpp
    Testsuite.cpp)

target_link_libraries(NetworkTests Base)
target_link_libraries(NetworkTests EngineInterface)
target_link_libraries(NetworkTests Network)

target_link_libraries(NetworkTests Boost::boost)
target_link_libraries(NetworkTests OpenGL::GL OpenGL::GLU)
target_link_libraries(NetworkTests GLEW::GLEW)
target_link_libraries(NetworkTests glfw)
target_link_libraries(NetworkTests glad::glad)
target_link_libraries(NetworkTests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(NetworkTests PRIVATE "/MP")
endif()
