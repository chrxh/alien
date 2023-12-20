target_sources(GuiTests
PUBLIC
    Testsuite.cpp)

target_link_libraries(GuiTests BaseLib)
target_link_libraries(GuiTests EngineGpuKernelsLib)
target_link_libraries(GuiTests EngineImplLib)
target_link_libraries(GuiTests EngineInterfaceLib)

target_link_libraries(GuiTests CUDA::cudart_static)
target_link_libraries(GuiTests CUDA::cuda_driver)
target_link_libraries(GuiTests Boost::boost)
target_link_libraries(GuiTests OpenGL::GL OpenGL::GLU)
target_link_libraries(GuiTests GLEW::GLEW)
target_link_libraries(GuiTests glfw)
target_link_libraries(GuiTests glad::glad)
target_link_libraries(GuiTests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(GuiTests PRIVATE "/MP")
endif()
