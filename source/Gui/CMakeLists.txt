
target_sources(alien
PUBLIC
    AboutDialog.cpp
    AboutDialog.h
    ActivateUserDialog.cpp
    ActivateUserDialog.h
    AlienDialog.cpp
    AlienDialog.h
    AlienImGui.cpp
    AlienImGui.h
    AlienWindow.cpp
    AlienWindow.h
    AutosaveController.cpp
    AutosaveController.h
    BrowserDataService.cpp
    BrowserDataService.h
    BrowserDataTO.cpp
    BrowserDataTO.h
    BrowserWindow.cpp
    BrowserWindow.h
    CellFunctionStrings.h
    ChangeColorDialog.cpp
    ChangeColorDialog.h
    CreateUserDialog.cpp
    CreateUserDialog.h
    CreatorWindow.cpp
    CreatorWindow.h
    Definitions.h
    DelayedExecutionController.cpp
    DelayedExecutionController.h
    DeleteUserDialog.cpp
    DeleteUserDialog.h
    DisplaySettingsDialog.cpp
    DisplaySettingsDialog.h
    EditorController.cpp
    EditorController.h
    EditorModel.cpp
    EditorModel.h
    ExitDialog.cpp
    ExitDialog.h
    FpsController.cpp
    FpsController.h
    GenericFileDialogs.cpp
    GenericFileDialogs.h
    GenomeEditorWindow.cpp
    GenomeEditorWindow.h
    GettingStartedWindow.cpp
    GettingStartedWindow.h
    GpuSettingsDialog.cpp
    GpuSettingsDialog.h
    GuiLogger.cpp
    GuiLogger.h
    HelpStrings.h
    ImageToPatternDialog.cpp
    ImageToPatternDialog.h
    InspectorWindow.cpp
    InspectorWindow.h
    LoginDialog.cpp
    LoginDialog.h
    LogWindow.cpp
    LogWindow.h
    Main.cpp
    MainWindow.cpp
    MainWindow.h
    MassOperationsDialog.cpp
    MassOperationsDialog.h
    MessageDialog.cpp
    MessageDialog.h
    ModeController.cpp
    ModeController.h
    MultiplierWindow.cpp
    MultiplierWindow.h
    NetworkController.cpp
    NetworkController.h
    NetworkDataParserService.cpp
    NetworkDataParserService.h
    NetworkDataTO.cpp
    NetworkDataTO.h
    NetworkSettingsDialog.cpp
    NetworkSettingsDialog.h
    NewSimulationDialog.cpp
    NewSimulationDialog.h
    NewPasswordDialog.cpp
    NewPasswordDialog.h
    OpenGLHelper.cpp
    OpenGLHelper.h
    OverlayMessageController.cpp
    OverlayMessageController.h
    PatternAnalysisDialog.cpp
    PatternAnalysisDialog.h
    PatternEditorWindow.cpp
    PatternEditorWindow.h
    RadiationSourcesWindow.cpp
    RadiationSourcesWindow.h
    ResetPasswordDialog.cpp
    ResetPasswordDialog.h
    ResizeWorldDialog.cpp
    ResizeWorldDialog.h
    SelectionWindow.cpp
    SelectionWindow.h
    Shader.cpp
    Shader.h
    ShaderWindow.cpp
    ShaderWindow.h
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
    StatisticsWindow.cpp
    StatisticsWindow.h
    StyleRepository.cpp
    StyleRepository.h
    TemporalControlWindow.cpp
    TemporalControlWindow.h
    TimelineLiveStatistics.cpp
    TimelineLiveStatistics.h
    UiController.cpp
    UiController.h
    UploadSimulationDialog.cpp
    UploadSimulationDialog.h
    UserTO.h
    Viewport.cpp
    Viewport.h
    WindowController.cpp
    WindowController.h)

target_link_libraries(alien BaseLib)
target_link_libraries(alien EngineGpuKernelsLib)
target_link_libraries(alien EngineImplLib)
target_link_libraries(alien EngineInterfaceLib)

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

if (MSVC)
    target_compile_options(alien PRIVATE "/MP")
endif()
