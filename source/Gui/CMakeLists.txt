
target_sources(alien
PUBLIC
    AboutDialog.cpp
    AboutDialog.h
    ActivateUserDialog.cpp
    ActivateUserDialog.h
    AlienImGui.cpp
    AlienImGui.h
    AlienWindow.cpp
    AlienWindow.h
    AutosaveController.cpp
    AutosaveController.h
    BrowserWindow.cpp
    BrowserWindow.h
    ColorizeDialog.cpp
    ColorizeDialog.h
    CreateUserDialog.cpp
    CreateUserDialog.h
    CreatorWindow.cpp
    CreatorWindow.h
    Definitions.h
    DeleteUserDialog.cpp
    DeleteUserDialog.h
    DisplaySettingsDialog.cpp
    DisplaySettingsDialog.h
    EditorController.cpp
    EditorController.h
    EditorModel.cpp
    EditorModel.h
    ExportStatisticsDialog.cpp
    ExportStatisticsDialog.h
    FileLogger.cpp
    FileLogger.h
    FlowGeneratorWindow.cpp
    FlowGeneratorWindow.h
    FpsController.cpp
    FpsController.h
    GettingStartedWindow.cpp
    GettingStartedWindow.h
    GlobalSettings.cpp
    GlobalSettings.h
    GpuSettingsDialog.cpp
    GpuSettingsDialog.h
    InspectorWindow.cpp
    InspectorWindow.h
    LoginDialog.cpp
    LoginDialog.h
    LogWindow.cpp
    LogWindow.h
    Main.cpp
    MainWindow.cpp
    MainWindow.h
    MessageDialog.cpp
    MessageDialog.h
    ModeController.cpp
    ModeController.h
    MultiplierWindow.cpp
    MultiplierWindow.h
    NetworkController.cpp
    NetworkController.h
    NetworkSettingsDialog.cpp
    NetworkSettingsDialog.h
    NewSimulationDialog.cpp
    NewSimulationDialog.h
    NewPasswordDialog.cpp
    NewPasswordDialog.h
    OpenGLHelper.cpp
    OpenGLHelper.h
    OpenPatternDialog.cpp
    OpenPatternDialog.h
    OpenSimulationDialog.cpp
    OpenSimulationDialog.h
    OpenSymbolsDialog.cpp
    OpenSymbolsDialog.h
    PatternAnalysisDialog.cpp
    PatternAnalysisDialog.h
    PatternEditorWindow.cpp
    PatternEditorWindow.h
    RemoteSimulationData.cpp
    RemoteSimulationData.h
    RemoteSimulationDataParser.cpp
    RemoteSimulationDataParser.h
    ResetPasswordDialog.cpp
    ResetPasswordDialog.h
    SavePatternDialog.cpp
    SavePatternDialog.h
    SaveSimulationDialog.cpp
    SaveSimulationDialog.h
    SaveSymbolsDialog.cpp
    SaveSymbolsDialog.h
    SelectionWindow.cpp
    SelectionWindow.h
    Shader.cpp
    Shader.h
    SimpleLogger.cpp
    SimpleLogger.h
    SimulationParametersCalculator.cpp
    SimulationParametersCalculator.h
    SimulationParametersChanger.cpp
    SimulationParametersChanger.h
    SimulationParametersWindow.cpp
    SimulationParametersWindow.h
    SimulationScrollbar.cpp
    SimulationScrollbar.h
    SimulationView.cpp
    SimulationView.h
    SpatialControlWindow.cpp
    SpatialControlWindow.h
    StartupController.cpp
    StartupController.h
    StatisticsHistory.cpp
    StatisticsHistory.h
    StatisticsWindow.cpp
    StatisticsWindow.h
    StyleRepository.cpp
    StyleRepository.h
    SymbolsWindow.cpp
    SymbolsWindow.h
    TemporalControlWindow.cpp
    TemporalControlWindow.h
    UiController.cpp
    UiController.h
    UploadSimulationDialog.cpp
    UploadSimulationDialog.h
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
target_link_libraries(alien OpenSSL::SSL OpenSSL::Crypto)
