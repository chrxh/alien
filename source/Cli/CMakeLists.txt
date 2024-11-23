target_sources(cli
PUBLIC
    Main.cpp)

target_link_libraries(cli Base)
target_link_libraries(cli EngineGpuKernels)
target_link_libraries(cli EngineImpl)
target_link_libraries(cli EngineInterface)
target_link_libraries(cli PersisterInterface)

target_link_libraries(cli CUDA::cudart_static)
target_link_libraries(cli CUDA::cuda_driver)
target_link_libraries(cli Boost::boost)
target_link_libraries(cli OpenGL::GL OpenGL::GLU)
target_link_libraries(cli GLEW::GLEW)
target_link_libraries(cli glfw)
target_link_libraries(cli glad::glad)
target_link_libraries(cli GTest::GTest GTest::Main)
target_link_libraries(cli glad::glad)
target_link_libraries(cli CLI11::CLI11)
target_link_libraries(cli ZLIB::ZLIB)

if (MSVC)
    target_compile_options(cli PRIVATE "/MP")
endif()
