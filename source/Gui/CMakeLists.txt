target_sources(alien
PUBLIC
    AboutDialog.cpp
    AboutDialog.h
    ActivateUserDialog.cpp
    ActivateUserDialog.h
    AlienDialog.cpp
    AlienDialog.h
    AlienGui.cpp
    AlienGui.h
    AlienWindow.cpp
    AlienWindow.h
    AutosaveWindow.cpp
    AutosaveWindow.h
    BrowserWindow.cpp
    BrowserWindow.h
    ChangeColorDialog.cpp
    ChangeColorDialog.h
    CreateUserDialog.cpp
    CreateUserDialog.h
    CreatorWindow.cpp
    CreatorWindow.h
    CreaturePreviewWidget.cpp
    CreaturePreviewWidget.h
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
    FileTransferController.cpp
    FileTransferController.h
    FpsController.cpp
    FpsController.h
    GenericFileDialog.cpp
    GenericFileDialog.h
    GenericMessageDialog.cpp
    GenericMessageDialog.h
    GeneEditorWidget.cpp
    GeneEditorWidget.h
    GenomeEditorWidget.cpp
    GenomeEditorWidget.h
    GenomeEditorWindow.cpp
    GenomeEditorWindow.h
    GenomeTabEditData.h
    GenomeTabLayoutData.h
    GenomeTabWidget.cpp
    GenomeTabWidget.h
    GenomeWindowEditData.h
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
    InspectionWindow.cpp
    InspectionWindow.h
    LastSessionBrowserData.cpp
    LastSessionBrowserData.h
    LayerColorPalette.cpp
    LayerColorPalette.h
    LocationController.cpp
    LocationController.h
    LocationWidget.h
    LocationWindow.cpp
    LocationWindow.h
    LoginController.cpp
    LoginController.h
    LoginDialog.cpp
    LoginDialog.h
    LogWindow.cpp
    LogWindow.h
    Main.cpp
    MainLoopController.cpp
    MainLoopController.h
    MainLoopEntity.cpp
    MainLoopEntity.h
    MainLoopEntityController.cpp
    MainLoopEntityController.h
    MainWindow.cpp
    MainWindow.h
    MassOperationsDialog.cpp
    MassOperationsDialog.h
    MultiplierWindow.cpp
    MultiplierWindow.h
    MutationRatesDialog.cpp
    MutationRatesDialog.h
    MutationRatesWidget.cpp
    MutationRatesWidget.h
    NetworkSettingsDialog.cpp
    NetworkSettingsDialog.h
    NetworkTransferController.cpp
    NetworkTransferController.h
    NeuralNetEditorWidget.cpp
    NeuralNetEditorWidget.h
    NewSimulationDialog.cpp
    NewSimulationDialog.h
    NewPasswordDialog.cpp
    NewPasswordDialog.h
    NodeEditorWidget.cpp
    NodeEditorWidget.h
    OpenGLHelper.cpp
    OpenGLHelper.h
    OverlayController.cpp
    OverlayController.h
    PatternEditorWindow.cpp
    PatternEditorWindow.h
    PreviewSettingsDialog.cpp
    PreviewSettingsDialog.h
    PreviewWidget.cpp
    PreviewWidget.h
    RenderPipeline.cpp
    RenderPipeline.h
    RenderStep.cpp
    RenderStep.h
    ResetPasswordDialog.cpp
    ResetPasswordDialog.h
    ResizeWorldDialog.cpp
    ResizeWorldDialog.h
    SelectionWindow.cpp
    SelectionWindow.h
    Shader.cpp
    Shader.h
    SignalsBufferDialog.cpp
    SignalsBufferDialog.h
    SimulationInteractionController.cpp
    SimulationInteractionController.h
    SimulationParametersBaseWidget.cpp
    SimulationParametersBaseWidget.h
    SimulationParametersMainWindow.cpp
    SimulationParametersMainWindow.h
    SimulationParametersSourceWidget.cpp
    SimulationParametersSourceWidget.h
    SimulationParametersLayerWidget.cpp
    SimulationParametersLayerWidget.h
    SimulationScrollbars.cpp
    SimulationScrollbars.h
    SimulationView.cpp
    SimulationView.h
    SpatialControlWindow.cpp
    SpatialControlWindow.h
    SpecificationGuiService.cpp
    SpecificationGuiService.h
    StartupCheckService.cpp
    StartupCheckService.h
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
target_link_libraries(alien EngineKernels)
target_link_libraries(alien EngineImpl)
target_link_libraries(alien EngineInterface)
target_link_libraries(alien Network)
target_link_libraries(alien PersisterInterface)
target_link_libraries(alien PersisterImpl)

target_include_directories(alien
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(alien cudart)
target_link_libraries(alien im_file_dialog)
target_link_libraries(alien Boost::boost)
target_link_libraries(alien OpenGL::GL OpenGL::GLU)
target_link_libraries(alien GLEW::GLEW)
target_link_libraries(alien imgui::imgui)
target_link_libraries(alien implot::implot)
target_link_libraries(alien glfw)
target_link_libraries(alien glad::glad)
target_link_libraries(alien OpenSSL::SSL OpenSSL::Crypto)

include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

target_precompile_headers(alien PRIVATE
    <algorithm>
    <chrono>
    <cmath>
    <filesystem>
    <functional>
    <map>
    <memory>
    <optional>
    <ranges>
    <sstream>
    <string>
    <unordered_map>
    <unordered_set>
    <variant>
    <vector>
    <imgui.h>
    <boost/algorithm/string.hpp>
)

if (MSVC)
    target_compile_options(alien PRIVATE "/MP")
endif()
