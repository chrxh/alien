
target_sources(alien
PUBLIC
    AboutDialog.cpp
    AboutDialog.h
    ManipulatorWindow.cpp
    ManipulatorWindow.h
    AlienImGui.cpp
    AlienImGui.h
    AlienWindow.cpp
    AlienWindow.h
    AutosaveController.cpp
    AutosaveController.h
    ColorizeDialog.cpp
    ColorizeDialog.h
    CreatorWindow.cpp
    CreatorWindow.h
    Definitions.h
    DisplaySettingsDialog.cpp
    DisplaySettingsDialog.h
    EditorController.cpp
    EditorController.h
    EditorModel.cpp
    EditorModel.h
    FileLogger.cpp
    FileLogger.h
    FlowGeneratorWindow.cpp
    FlowGeneratorWindow.h
    GettingStartedWindow.cpp
    GettingStartedWindow.h
    GlobalSettings.cpp
    GlobalSettings.h
    GpuSettingsDialog.cpp
    GpuSettingsDialog.h
    InspectorWindow.cpp
    InspectorWindow.h
    LogWindow.cpp
    LogWindow.h
    Main.cpp
    MainWindow.cpp
    MainWindow.h
    ModeController.cpp
    ModeController.h
    MultiplierWindow.cpp
    MultiplierWindow.h
    NewSimulationDialog.cpp
    NewSimulationDialog.h
    OpenGLHelper.cpp
    OpenGLHelper.h
    OpenSelectionDialog.cpp
    OpenSelectionDialog.h
    OpenSimulationDialog.cpp
    OpenSimulationDialog.h
    Resources.h
    SaveSelectionDialog.cpp
    SaveSelectionDialog.h
    SaveSimulationDialog.cpp
    SaveSimulationDialog.h
    SelectionWindow.cpp
    SelectionWindow.h
    Shader.cpp
    Shader.h
    SimpleLogger.cpp
    SimpleLogger.h
    SimulationParametersWindow.cpp
    SimulationParametersWindow.h
    SimulationScrollbar.cpp
    SimulationScrollbar.h
    SimulationView.cpp
    SimulationView.h
    SpatialControlWindow.cpp
    SpatialControlWindow.h
    StartupWindow.cpp
    StartupWindow.h
    StatisticsWindow.cpp
    StatisticsWindow.h
    StyleRepository.cpp
    StyleRepository.h
    TemporalControlWindow.cpp
    TemporalControlWindow.h
    UiController.cpp
    UiController.h
    Viewport.cpp
    Viewport.h
    WindowController.cpp
    WindowController.h)

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
