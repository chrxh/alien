target_sources(gui_tests
PUBLIC
    Testsuite.cpp)

target_link_libraries(gui_tests alien_base_lib)
target_link_libraries(gui_tests alien_engine_gpu_kernels_lib)
target_link_libraries(gui_tests alien_engine_impl_lib)
target_link_libraries(gui_tests alien_engine_interface_lib)

target_link_libraries(gui_tests CUDA::cudart_static)
target_link_libraries(gui_tests CUDA::cuda_driver)
target_link_libraries(gui_tests Boost::boost)
target_link_libraries(gui_tests OpenGL::GL OpenGL::GLU)
target_link_libraries(gui_tests GLEW::GLEW)
target_link_libraries(gui_tests glfw)
target_link_libraries(gui_tests glad::glad)
target_link_libraries(gui_tests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(gui_tests PRIVATE "/MP")
endif()
