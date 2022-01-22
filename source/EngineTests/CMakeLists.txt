
target_sources(tests
PUBLIC
    main.cpp)

target_link_libraries(alien alien_base_lib)
target_link_libraries(alien alien_engine_gpu_kernels_lib)
target_link_libraries(alien alien_engine_impl_lib)
target_link_libraries(alien alien_engine_interface_lib)
target_link_libraries(alien im_file_dialog)

target_link_libraries(alien CUDA::cudart_static)
target_link_libraries(alien CUDA::cuda_driver)
target_link_libraries(alien Boost::boost)
target_link_libraries(alien OpenGL::GL OpenGL::GLU)
target_link_libraries(alien GLEW::GLEW)
target_link_libraries(alien imgui::imgui)
target_link_libraries(alien implot::implot)
target_link_libraries(alien glfw)
target_link_libraries(alien glad::glad)
