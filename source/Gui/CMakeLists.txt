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
    AutosaveController.cpp
    AutosaveController.h
    AutosaveWindow.cpp
    AutosaveWindow.h
    BrowserController.cpp
    BrowserController.h
    BrowserData.cpp
    BrowserData.h
    BrowserGalleryWidget.cpp
    BrowserGalleryWidget.h
    BrowserHelper.cpp
    BrowserHelper.h
    BrowserLoginBannerWidget.cpp
    BrowserLoginBannerWidget.h
    BrowserLoginHintWidget.cpp
    BrowserLoginHintWidget.h
    BrowserTableWidget.cpp
    BrowserTableWidget.h
    BrowserUserListWidget.cpp
    BrowserUserListWidget.h
    BrowserWindow.cpp
    BrowserWindow.h
    CellFunctionChannels.cpp
    CellFunctionChannels.h
    ChangeColorDialog.cpp
    ChangeColorDialog.h
    ColorMatrixDialog.cpp
    ColorMatrixDialog.h
    ConsoleModeController.cpp
    ConsoleModeController.h
    CreateUserDialog.cpp
    CreateUserDialog.h
    CreatorController.cpp
    CreatorController.h
    CreatorWidget.cpp
    CreatorWidget.h
    CreaturePreviewWidget.cpp
    CreaturePreviewWidget.h
    Definitions.h
    DelayedExecutionController.cpp
    DelayedExecutionController.h
    DeleteUserDialog.cpp
    DeleteUserDialog.h
    DisplaySettingsDialog.cpp
    DisplaySettingsDialog.h
    DocumentationWindow.cpp
    DocumentationWindow.h
    EditInteractionController.h
    EditorController.cpp
    EditorController.h
    EditorModel.cpp
    EditorModel.h
    EditorWidget.cpp
    EditorWidget.h
    EditSimulationDialog.cpp
    EditSimulationDialog.h
    EntityAttributeHelp.cpp
    EntityAttributeHelp.h
    EvolutionDashboardWindow.cpp
    EvolutionDashboardWindow.h
    ExitDialog.cpp
    ExitDialog.h
    FileTransferController.cpp
    FileTransferController.h
    ForceController.cpp
    ForceController.h
    FpsController.cpp
    FpsController.h
    GenericFileDialog.cpp
    GenericFileDialog.h
    GenericMessageDialog.cpp
    GenericMessageDialog.h
    GeneEditorWidget.cpp
    GeneEditorWidget.h
    GeneGraphWidget.cpp
    GeneGraphWidget.h
    GenomeEditorWidget.cpp
    GenomeEditorWidget.h
    GenomeEditorWindow.cpp
    GenomeEditorWindow.h
    GenomeIssueDescription.cpp
    GenomeIssueDescription.h
    GenomeTabEditData.h
    GenomeTabLayoutData.h
    GenomeTabWidget.cpp
    GenomeTabWidget.h
    GenomeWindowEditData.h
    GuiLogger.cpp
    GuiLogger.h
    HelpStrings.h
    ImageToPatternDialog.cpp
    ImageToPatternDialog.h
    InspectionController.cpp
    InspectionController.h
    InspectionWindow.cpp
    InspectionWindow.h
    LastSessionBrowserData.cpp
    LastSessionBrowserData.h
    LayerColorPalette.cpp
    LayerColorPalette.h
    LiveStatisticsHistory.cpp
    LiveStatisticsHistory.h
    LiveStatisticsService.cpp
    LiveStatisticsService.h
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
    MarkdownRenderer.cpp
    MarkdownRenderer.h
    MassOperationsDialog.cpp
    MassOperationsDialog.h
    McpController.cpp
    McpController.h
    McpSettingsDialog.cpp
    McpSettingsDialog.h
    McpWindow.cpp
    McpWindow.h
    ModalWindow.cpp
    ModalWindow.h
    MultiplierWidget.cpp
    MultiplierWidget.h
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
    NewSimulationService.cpp
    NewSimulationService.h
    NewPasswordDialog.cpp
    NewPasswordDialog.h
    NodeEditorWidget.cpp
    NodeEditorWidget.h
    OverlayController.cpp
    OverlayController.h
    PictureData.h
    PictureGuiService.cpp
    PictureGuiService.h
    PreviewDescView.cpp
    PreviewDescView.h
    PreviewSettingsDialog.cpp
    PreviewSettingsDialog.h
    PreviewViewport.cpp
    PreviewViewport.h
    PreviewWidget.cpp
    PreviewWidget.h
    RenderGraph.cpp
    RenderGraph.h
    RenderStep.cpp
    RenderStep.h
    ReplaceSimulationDialog.cpp
    ReplaceSimulationDialog.h
    ResetPasswordDialog.cpp
    ResetPasswordDialog.h
    ResizeWorldDialog.cpp
    ResizeWorldDialog.h
    ResourcePreviewWidget.cpp
    ResourcePreviewWidget.h
    SavePictureDialog.cpp
    SavePictureDialog.h
    ScissorsController.cpp
    ScissorsController.h
    SelectionController.cpp
    SelectionController.h
    SelectionWidget.cpp
    SelectionWidget.h
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
    StyleService.cpp
    StyleService.h
    TemporalControlWindow.cpp
    TemporalControlWindow.h
    TextureService.cpp
    TextureService.h
    UiController.cpp
    UiController.h
    UploadSimulationDialog.cpp
    UploadSimulationDialog.h
    Viewport.cpp
    Viewport.h
    VulkanContext.cpp
    VulkanContext.h
    VulkanFrameRenderer.cpp
    VulkanFrameRenderer.h
    VulkanGeometryBuffers.cpp
    VulkanGeometryBuffers.h
    WindowController.cpp
    WindowController.h)

target_link_libraries(alien Base)
target_link_libraries(alien ConsoleUi)
target_link_libraries(alien EngineKernels)
target_link_libraries(alien EngineImpl)
target_link_libraries(alien Data)
target_link_libraries(alien EngineInterface)
target_link_libraries(alien McpToolsImpl)
target_link_libraries(alien McpToolsInterface)
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
target_link_libraries(alien Vulkan::Vulkan)
target_link_libraries(alien glslang::glslang glslang::glslang-default-resource-limits glslang::SPIRV)
target_link_libraries(alien imgui::imgui)
target_link_libraries(alien implot::implot)
target_link_libraries(alien glfw)
target_link_libraries(alien OpenSSL::SSL OpenSSL::Crypto)

# GLFW must not pull in the OpenGL headers
target_compile_definitions(alien PRIVATE GLFW_INCLUDE_NONE)

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

# Embed the application icon. The path is baked into the generated resource script because
# the resource compiler resolves relative paths against the build directory, not the source one.
if (WIN32)
    set(ALIEN_ICON_FILE "${CMAKE_SOURCE_DIR}/resources/images/alien.ico")
    configure_file(Resources.rc.in ${CMAKE_CURRENT_BINARY_DIR}/Resources.rc @ONLY)
    target_sources(alien PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Resources.rc)
endif()
