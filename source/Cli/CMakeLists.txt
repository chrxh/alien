target_sources(cli
PUBLIC
    Main.cpp)

target_link_libraries(cli alien_base_lib)
target_link_libraries(cli alien_engine_gpu_kernels_lib)
target_link_libraries(cli alien_engine_impl_lib)
target_link_libraries(cli alien_engine_interface_lib)

target_link_libraries(cli CUDA::cudart_static)
target_link_libraries(cli CUDA::cuda_driver)
target_link_libraries(cli Boost::boost)
target_link_libraries(cli OpenGL::GL OpenGL::GLU)
target_link_libraries(cli GLEW::GLEW)
target_link_libraries(cli glfw)
target_link_libraries(cli glad::glad)
target_link_libraries(cli GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(cli PRIVATE "/MP")
endif()
