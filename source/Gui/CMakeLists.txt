
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
    AutosaveWindow.cpp
    AutosaveWindow.h
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
    EditSimulationDialog.cpp
    EditSimulationDialog.h
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
    HistogramLiveStatistics.cpp
    HistogramLiveStatistics.h
    ImageToPatternDialog.cpp
    ImageToPatternDialog.h
    InspectorWindow.cpp
    InspectorWindow.h
    LastSessionBrowserData.cpp
    LastSessionBrowserData.h
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
    MultiplierWindow.cpp
    MultiplierWindow.h
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
    PersisterJob.cpp
    PersisterJob.h
    PersisterWorker.cpp
    PersisterWorker.h
    RadiationSourcesWindow.cpp
    RadiationSourcesWindow.h
    ResetPasswordDialog.cpp
    ResetPasswordDialog.h
    ResizeWorldDialog.cpp
    ResizeWorldDialog.h
    SelectionWindow.cpp
    SelectionWindow.h
    SerializationHelperService.cpp
    SerializationHelperService.h
    Shader.cpp
    Shader.h
    ShaderWindow.cpp
    ShaderWindow.h
    SimulationInteractionController.cpp
    SimulationInteractionController.h
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
    TableLiveStatistics.cpp
    TableLiveStatistics.h
    TemporalControlWindow.cpp
    TemporalControlWindow.h
    TimelineLiveStatistics.cpp
    TimelineLiveStatistics.h
    UiController.cpp
    UiController.h
    UploadSimulationDialog.cpp
    UploadSimulationDialog.h
    Viewport.cpp
    Viewport.h
    WindowController.cpp
    WindowController.h)

target_link_libraries(alien Base)
target_link_libraries(alien EngineGpuKernels)
target_link_libraries(alien EngineImpl)
target_link_libraries(alien EngineInterface)
target_link_libraries(alien Network)

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
