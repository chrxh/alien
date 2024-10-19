#include "MainWindow.h"

#include <iostream>

#include <glad/glad.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "ImFileDialog.h"
#include "implot.h"
#include "Fonts/IconsFontAwesome5.h"

#include "PersisterInterface/PersisterFacade.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationFacade.h"
#include "Network/NetworkService.h"

#include "SimulationInteractionController.h"
#include "SimulationView.h"
#include "StyleRepository.h"
#include "TemporalControlWindow.h"
#include "SpatialControlWindow.h"
#include "SimulationParametersWindow.h"
#include "StatisticsWindow.h"
#include "GpuSettingsDialog.h"
#include "Viewport.h"
#include "NewSimulationDialog.h"
#include "StartupController.h"
#include "AlienImGui.h"
#include "AboutDialog.h"
#include "MassOperationsDialog.h"
#include "LogWindow.h"
#include "GuiLogger.h"
#include "UiController.h"
#include "AutosaveController.h"
#include "GettingStartedWindow.h"
#include "DisplaySettingsDialog.h"
#include "EditorController.h"
#include "SelectionWindow.h"
#include "PatternEditorWindow.h"
#include "WindowController.h"
#include "CreatorWindow.h"
#include "MultiplierWindow.h"
#include "PatternAnalysisDialog.h"
#include "MessageDialog.h"
#include "FpsController.h"
#include "BrowserWindow.h"
#include "LoginDialog.h"
#include "UploadSimulationDialog.h"
#include "EditSimulationDialog.h"
#include "CreateUserDialog.h"
#include "ActivateUserDialog.h"
#include "DelayedExecutionController.h"
#include "DeleteUserDialog.h"
#include "NetworkSettingsDialog.h"
#include "ResetPasswordDialog.h"
#include "NewPasswordDialog.h"
#include "ImageToPatternDialog.h"
#include "GenericFileDialogs.h"
#include "ShaderWindow.h"
#include "GenomeEditorWindow.h"
#include "RadiationSourcesWindow.h"
#include "OverlayMessageController.h"
#include "ExitDialog.h"
#include "AutosaveWindow.h"
#include "FileTransferController.h"
#include "LoginController.h"
#include "NetworkTransferController.h"

namespace
{
    void glfwErrorCallback(int error, const char* description)
    {
        throw std::runtime_error("Glfw error " + std::to_string(error) + ": " + description);
    }

    void framebuffer_size_callback(GLFWwindow* window, int width, int height)
    {
        if (width > 0 && height > 0) {
            SimulationView::get().resize({width, height});
            glViewport(0, 0, width, height);
        }
    }
}

_MainWindow::_MainWindow(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade, GuiLogger const& logger)
    : _logger(logger)
    , _simulationFacade(simulationFacade)
    , _persisterFacade(persisterFacade)
{
    IMGUI_CHECKVERSION();

    log(Priority::Important, "initialize GLFW and OpenGL");
    initGlfwAndOpenGL();

    log(Priority::Important, "initialize GLAD");
    initGlad();

    //init services
    StyleRepository::get().init();
    NetworkService::get().init();

    //init facades
    _persisterFacade->init(_simulationFacade);

    //init controllers, windows and dialogs
    Viewport::get().init(_simulationFacade);
    AutosaveController::get().init(_simulationFacade);
    EditorController::get().init(_simulationFacade);
    SimulationView::get().init(_simulationFacade);
    SimulationInteractionController::get().init(_simulationFacade);
    StatisticsWindow::get().init(_simulationFacade);
    TemporalControlWindow::get().init(_simulationFacade);
    SpatialControlWindow::get().init(_simulationFacade);
    RadiationSourcesWindow::get().init(_simulationFacade);
    SimulationParametersWindow::get().init(_simulationFacade);
    _gpuSettingsDialog = std::make_shared<_GpuSettingsDialog>(_simulationFacade);
    _startupController = std::make_shared<_StartupController>(_simulationFacade, _persisterFacade);
    _exitDialog = std::make_shared<_ExitDialog>(_onExit);
    _aboutDialog = std::make_shared<_AboutDialog>();
    _massOperationsDialog = std::make_shared<_MassOperationsDialog>(_simulationFacade);
    _logWindow = std::make_shared<_LogWindow>(_logger);
    _gettingStartedWindow = std::make_shared<_GettingStartedWindow>();
    _newSimulationDialog = std::make_shared<_NewSimulationDialog>(_simulationFacade);
    _displaySettingsDialog = std::make_shared<_DisplaySettingsDialog>();
    _patternAnalysisDialog = std::make_shared<_PatternAnalysisDialog>(_simulationFacade);
    _fpsController = std::make_shared<_FpsController>();
    BrowserWindow::get().init(_simulationFacade, _persisterFacade);
    ActivateUserDialog::get().init(_simulationFacade);
    _newPasswordDialog = std::make_shared<_NewPasswordDialog>(_simulationFacade);
    _resetPasswordDialog = std::make_shared<_ResetPasswordDialog>(_newPasswordDialog);
    LoginDialog::get().init(_simulationFacade, _persisterFacade, _resetPasswordDialog);
    UploadSimulationDialog::get().init(_simulationFacade);
    _deleteUserDialog = std::make_shared<_DeleteUserDialog>();
    _networkSettingsDialog = std::make_shared<_NetworkSettingsDialog>();
    _imageToPatternDialog = std::make_shared<_ImageToPatternDialog>(_simulationFacade);
    _shaderWindow = std::make_shared<_ShaderWindow>();
    _autosaveWindow = std::make_shared<_AutosaveWindow>(_simulationFacade, _persisterFacade);
    OverlayMessageController::get().init(_persisterFacade);
    FileTransferController::get().init(_persisterFacade, _simulationFacade);
    NetworkTransferController::get().init(_simulationFacade, _persisterFacade);
    LoginController::get().init(_simulationFacade, _persisterFacade);

    ifd::FileDialog::Instance().CreateTexture = [](uint8_t* data, int w, int h, char fmt) -> void* {
        GLuint tex;

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, (fmt == 0) ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        return reinterpret_cast<void*>(uintptr_t(tex));
    };
    ifd::FileDialog::Instance().DeleteTexture = [](void* tex) {
        GLuint texID = reinterpret_cast<uintptr_t>(tex);
        glDeleteTextures(1, &texID);
    };

    log(Priority::Important, "main window initialized");
}

void _MainWindow::mainLoop()
{
    auto window = WindowController::get().getWindowData().window;
    while (!glfwWindowShouldClose(window) && !_onExit)
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

     //   ImGui::ShowDemoWindow(NULL);

        switch (_startupController->getState()) {
        case _StartupController::State::StartLoadSimulation:
            processLoadingScreen();
            break;
        case _StartupController::State::LoadingSimulation:
            processLoadingScreen();
            break;
        case _StartupController::State::FadeOutLoadingScreen:
            processFadeoutLoadingScreen();
            break;
        case _StartupController::State::FadeInUI:
            processFadeInUI();
            break;
        case _StartupController::State::Ready:
            processReady();
            break;
        default:
            THROW_NOT_IMPLEMENTED();
        }
    }
}

void _MainWindow::shutdown()
{
    BrowserWindow::get().shutdown();
    StatisticsWindow::get().shutdown();
    SpatialControlWindow::get().shutdown();
    SimulationParametersWindow::get().shutdown();

    EditorController::get().shutdown();
    LoginController::get().shutdown();
    WindowController::get().shutdown();
    AutosaveController::get().shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(WindowController::get().getWindowData().window);
    glfwTerminate();

    SimulationView::get().shutdown();

    _persisterFacade->shutdown();
    _simulationFacade->closeSimulation();
    NetworkService::get().shutdown();
}

void _MainWindow::initGlfwAndOpenGL()
{
    auto glfwVersion = initGlfwAndReturnGlslVersion();
    WindowController::get().init();
    auto windowData = WindowController::get().getWindowData();
    glfwSetFramebufferSizeCallback(windowData.window, framebuffer_size_callback);
    glfwSwapInterval(1);  //enable vsync
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(windowData.window, true);  //setup Platform/Renderer back-ends
    ImGui_ImplOpenGL3_Init(glfwVersion);
}

void _MainWindow::initGlad()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
}

char const* _MainWindow::initGlfwAndReturnGlslVersion()
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize Glfw.");
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glslVersion = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    return glslVersion;
}

void _MainWindow::processLoadingScreen()
{
    _startupController->process();
    OverlayMessageController::get().process();

    // render mainData
    auto window = WindowController::get().getWindowData().window;
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
}

void _MainWindow::processFadeoutLoadingScreen()
{
    _startupController->process();
    renderSimulation();

    finishFrame();
}

void _MainWindow::processFadeInUI()
{
    renderSimulation();

    pushGlobalStyle();

    processMenubar();
    processDialogs();
    processWindows();
    processControllers();

    SimulationView::get().processControls(_renderSimulation);
    _startupController->process();

    popGlobalStyle();

    _fpsController->processForceFps(WindowController::get().getFps());

    finishFrame();
}

void _MainWindow::processReady()
{
    renderSimulation();

    pushGlobalStyle();

    processMenubar();
    processDialogs();
    processWindows();
    processControllers();

    SimulationView::get().processControls(_renderSimulation);

    popGlobalStyle();

    _fpsController->processForceFps(WindowController::get().getFps());

    finishFrame();
}

void _MainWindow::renderSimulation()
{
    int display_w, display_h;
    glfwGetFramebufferSize(WindowController::get().getWindowData().window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    SimulationView::get().draw(_renderSimulation);
}

void _MainWindow::processMenubar()
{
    auto selectionWindow = EditorController::get().getSelectionWindow();
    auto patternEditorWindow = EditorController::get().getPatternEditorWindow();
    auto creatorWindow = EditorController::get().getCreatorWindow();
    auto multiplierWindow = EditorController::get().getMultiplierWindow();
    auto genomeEditorWindow = EditorController::get().getGenomeEditorWindow();

    if (ImGui::BeginMainMenuBar()) {
        if (AlienImGui::ShutdownButton()) {
            onExit();
        }
        ImGui::Dummy(ImVec2(10.0f, 0.0f));
        if (AlienImGui::BeginMenuButton(" " ICON_FA_GAMEPAD "  Simulation ", _simulationMenuToggled, "Simulation")) {
            if (ImGui::MenuItem("New", "CTRL+N")) {
                _newSimulationDialog->open();
                _simulationMenuToggled = false;
            }
            if (ImGui::MenuItem("Open", "CTRL+O")) {
                FileTransferController::get().onOpenSimulation();
                _simulationMenuToggled = false;
            }
            if (ImGui::MenuItem("Save", "CTRL+S")) {
                FileTransferController::get().onSaveSimulation();
                _simulationMenuToggled = false;
            }
            ImGui::Separator();
            ImGui::BeginDisabled(_simulationFacade->isSimulationRunning());
            if (ImGui::MenuItem("Run", "SPACE")) {
                onRunSimulation();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!_simulationFacade->isSimulationRunning());
            if (ImGui::MenuItem("Pause", "SPACE")) {
                onPauseSimulation();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_GLOBE "  Network ", _networkMenuToggled, "Network", false)) {
            if (ImGui::MenuItem("Browser", "ALT+W", BrowserWindow::get().isOn())) {
                BrowserWindow::get().setOn(!BrowserWindow::get().isOn());
            }
            ImGui::Separator();
            ImGui::BeginDisabled((bool)NetworkService::get().getLoggedInUserName());
            if (ImGui::MenuItem("Login", "ALT+L")) {
                LoginDialog::get().open();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!NetworkService::get().getLoggedInUserName());
            if (ImGui::MenuItem("Logout", "ALT+T")) {
                NetworkService::get().logout();
                BrowserWindow::get().onRefresh();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!NetworkService::get().getLoggedInUserName());
            if (ImGui::MenuItem("Upload simulation", "ALT+D")) {
                UploadSimulationDialog::get().open(NetworkResourceType_Simulation);
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!NetworkService::get().getLoggedInUserName());
            if (ImGui::MenuItem("Upload genome", "ALT+Q")) {
                UploadSimulationDialog::get().open(NetworkResourceType_Genome);
            }
            ImGui::EndDisabled();

            ImGui::Separator();
            ImGui::BeginDisabled(!NetworkService::get().getLoggedInUserName());
            if (ImGui::MenuItem("Delete user", "ALT+J")) {
                _deleteUserDialog->open();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_WINDOW_RESTORE "  Windows ", _windowMenuToggled, "Windows")) {
            if (ImGui::MenuItem("Temporal control", "ALT+1", TemporalControlWindow::get().isOn())) {
                TemporalControlWindow::get().setOn(!TemporalControlWindow::get().isOn());
            }
            if (ImGui::MenuItem("Spatial control", "ALT+2", SpatialControlWindow::get().isOn())) {
                SpatialControlWindow::get().setOn(!SpatialControlWindow::get().isOn());
            }
            if (ImGui::MenuItem("Statistics", "ALT+3", StatisticsWindow::get().isOn())) {
                StatisticsWindow::get().setOn(!StatisticsWindow::get().isOn());
            }
            if (ImGui::MenuItem("Simulation parameters", "ALT+4", SimulationParametersWindow::get().isOn())) {
                SimulationParametersWindow::get().setOn(!SimulationParametersWindow::get().isOn());
            }
            if (ImGui::MenuItem("Radiation sources", "ALT+5", RadiationSourcesWindow::get().isOn())) {
                RadiationSourcesWindow::get().setOn(!RadiationSourcesWindow::get().isOn());
            }
            if (ImGui::MenuItem("Shader parameters", "ALT+6", _shaderWindow->isOn())) {
                _shaderWindow->setOn(!_shaderWindow->isOn());
            }
            if (ImGui::MenuItem("Autosave", "ALT+7", _autosaveWindow->isOn())) {
                _autosaveWindow->setOn(!_autosaveWindow->isOn());
            }
            if (ImGui::MenuItem("Log", "ALT+8", _logWindow->isOn())) {
                _logWindow->setOn(!_logWindow->isOn());
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_PEN_ALT "  Editor ", _editorMenuToggled, "Editor")) {
            if (ImGui::MenuItem("Activate", "ALT+E", SimulationInteractionController::get().isEditMode())) {
                SimulationInteractionController::get().setEditMode(!SimulationInteractionController::get().isEditMode());
            }
            ImGui::Separator();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode());
            if (ImGui::MenuItem("Selection", "ALT+S", selectionWindow->isOn())) {
                selectionWindow->setOn(!selectionWindow->isOn());
            }
            if (ImGui::MenuItem("Creator", "ALT+R", creatorWindow->isOn())) {
                creatorWindow->setOn(!creatorWindow->isOn());
            }
            if (ImGui::MenuItem("Pattern editor", "ALT+M", patternEditorWindow->isOn())) {
                patternEditorWindow->setOn(!patternEditorWindow->isOn());
            }
            if (ImGui::MenuItem("Genome editor", "ALT+B", genomeEditorWindow->isOn())) {
                genomeEditorWindow->setOn(!genomeEditorWindow->isOn());
            }
            if (ImGui::MenuItem("Multiplier", "ALT+A", multiplierWindow->isOn())) {
                multiplierWindow->setOn(!multiplierWindow->isOn());
            }
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isObjectInspectionPossible());
            if (ImGui::MenuItem("Inspect objects", "ALT+N")) {
                EditorController::get().onInspectSelectedObjects();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isGenomeInspectionPossible());
            if (ImGui::MenuItem("Inspect principal genome", "ALT+F")) {
                EditorController::get().onInspectSelectedGenomes();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().areInspectionWindowsActive());
            if (ImGui::MenuItem("Close inspections", "ESC")) {
                EditorController::get().onCloseAllInspectorWindows();
            }
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isCopyingPossible());
            if (ImGui::MenuItem("Copy", "CTRL+C")) {
                EditorController::get().onCopy();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isPastingPossible());
            if (ImGui::MenuItem("Paste", "CTRL+V")) {
                EditorController::get().onPaste();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_EYE "  View ", _viewMenuToggled, "View")) {
            if (ImGui::MenuItem("Information overlay", "ALT+O", SimulationView::get().isOverlayActive())) {
                SimulationView::get().setOverlayActive(!SimulationView::get().isOverlayActive());
            }
            if (ImGui::MenuItem("Render UI", "ALT+U", UiController::get().isOn())) {
                UiController::get().setOn(!UiController::get().isOn());
            }
            if (ImGui::MenuItem("Render simulation", "ALT+I", _renderSimulation)) {
                _renderSimulation = !_renderSimulation;
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_TOOLS "  Tools ", _toolsMenuToggled, "Tools")) {
            if (ImGui::MenuItem("Mass operations", "ALT+H")) {
                _massOperationsDialog->show();
                _toolsMenuToggled = false;
            }
            if (ImGui::MenuItem("Pattern analysis", "ALT+P")) {
                _patternAnalysisDialog->show();
                _toolsMenuToggled = false;
            }
            if (ImGui::MenuItem("Image converter", "ALT+G")) {
                _imageToPatternDialog->show();
                _toolsMenuToggled = false;
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_COG "  Settings ", _settingsMenuToggled, "Settings", false)) {
            if (ImGui::MenuItem("Auto save", "", AutosaveController::get().isOn())) {
                AutosaveController::get().setOn(!AutosaveController::get().isOn());
            }
            if (ImGui::MenuItem("CUDA settings", "ALT+C")) {
                _gpuSettingsDialog->open();
            }
            if (ImGui::MenuItem("Display settings", "ALT+V")) {
                _displaySettingsDialog->open();
            }
            if (ImGui::MenuItem("Network settings", "ALT+K")) {
                _networkSettingsDialog->open();
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_LIFE_RING "  Help ", _helpMenuToggled, "Help")) {
            if (ImGui::MenuItem("About", "")) {
                _aboutDialog->open();
                _helpMenuToggled = false;
            }
            if (ImGui::MenuItem("Getting started", "", _gettingStartedWindow->isOn())) {
                _gettingStartedWindow->setOn(!_gettingStartedWindow->isOn());
            }
            AlienImGui::EndMenuButton();
        }
        ImGui::EndMainMenuBar();
    }

    //hotkeys
    auto& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_N)) {
            _newSimulationDialog->open();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O)) {
            FileTransferController::get().onOpenSimulation();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) {
            FileTransferController::get().onSaveSimulation();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
            if (_simulationFacade->isSimulationRunning()) {
                onPauseSimulation();
            } else {
                onRunSimulation();
            }
            
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_W)) {
            BrowserWindow::get().setOn(!BrowserWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_L) && !NetworkService::get().getLoggedInUserName()) {
            LoginDialog::get().open();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_T)) {
            NetworkService::get().logout();
            BrowserWindow::get().onRefresh();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_D) && NetworkService::get().getLoggedInUserName()) {
            UploadSimulationDialog::get().open(NetworkResourceType_Simulation);
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_Q) && NetworkService::get().getLoggedInUserName()) {
            UploadSimulationDialog::get().open(NetworkResourceType_Genome);
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_J) && NetworkService::get().getLoggedInUserName()) {
            _deleteUserDialog->open();
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_1)) {
            TemporalControlWindow::get().setOn(!TemporalControlWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_2)) {
            SpatialControlWindow::get().setOn(!SpatialControlWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_3)) {
            StatisticsWindow::get().setOn(!StatisticsWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_4)) {
            SimulationParametersWindow::get().setOn(!SimulationParametersWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_5)) {
            RadiationSourcesWindow::get().setOn(!RadiationSourcesWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_6)) {
            _shaderWindow->setOn(!_shaderWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_7)) {
            _autosaveWindow->setOn(!_autosaveWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_8)) {
            _logWindow->setOn(!_logWindow->isOn());
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_E)) {
            SimulationInteractionController::get().setEditMode(!SimulationInteractionController::get().isEditMode());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_S)) {
            selectionWindow->setOn(!selectionWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_M)) {
            patternEditorWindow->setOn(!patternEditorWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_B)) {
            genomeEditorWindow->setOn(!genomeEditorWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_R)) {
            creatorWindow->setOn(!creatorWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_A)) {
            multiplierWindow->setOn(!multiplierWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_N) && EditorController::get().isObjectInspectionPossible()) {
            EditorController::get().onInspectSelectedObjects();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_F) && EditorController::get().isGenomeInspectionPossible()) {
            EditorController::get().onInspectSelectedGenomes();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            EditorController::get().onCloseAllInspectorWindows();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C) && EditorController::get().isCopyingPossible()) {
            EditorController::get().onCopy();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V) && EditorController::get().isPastingPossible()) {
            EditorController::get().onPaste();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) ) {
            EditorController::get().onDelete();
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_C)) {
            _gpuSettingsDialog->open();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_V)) {
            _displaySettingsDialog->open();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F7)) {
            if (WindowController::get().isDesktopMode()) {
                WindowController::get().setWindowedMode();
            } else {
                WindowController::get().setDesktopMode();
            }
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_K)) {
            _networkSettingsDialog->open();
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_O)) {
            SimulationView::get().setOverlayActive(!SimulationView::get().isOverlayActive());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_U)) {
            UiController::get().setOn(!UiController::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_I)) {
            _renderSimulation = !_renderSimulation;
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_H)) {
            _massOperationsDialog->show();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_P)) {
            _patternAnalysisDialog->show();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_G)) {
            _imageToPatternDialog->show();
        }
    }
}

void _MainWindow::processDialogs()
{
    _newSimulationDialog->process();
    _aboutDialog->process();
    _massOperationsDialog->process();
    _gpuSettingsDialog->process();
    _displaySettingsDialog->process(); 
    _patternAnalysisDialog->process();
    LoginDialog::get().process();
    CreateUserDialog::get().process();
    ActivateUserDialog::get().process();
    UploadSimulationDialog::get().process();
    EditSimulationDialog::get().process();
    _deleteUserDialog->process();
    _networkSettingsDialog->process();
    _resetPasswordDialog->process();
    _newPasswordDialog->process();
    _exitDialog->process();

    MessageDialog::get().process();
    GenericFileDialogs::get().process();
}

void _MainWindow::processWindows()
{
    TemporalControlWindow::get().process();
    SpatialControlWindow::get().process();
    StatisticsWindow::get().process();
    SimulationParametersWindow::get().process();
    _logWindow->process();
    BrowserWindow::get().process();
    _gettingStartedWindow->process();
    _shaderWindow->process();
    RadiationSourcesWindow::get().process();
    _autosaveWindow->process();
}

void _MainWindow::processControllers()
{
    AutosaveController::get().process();
    EditorController::get().process();
    OverlayMessageController::get().process();
    SimulationInteractionController::get().process();
    DelayedExecutionController::get().process();
    FileTransferController::get().process();
    NetworkTransferController::get().process();
    LoginController::get().process();
    UiController::get().process();
}

void _MainWindow::onRunSimulation()
{
    _simulationFacade->runSimulation();
    printOverlayMessage("Run");
}

void _MainWindow::onPauseSimulation()
{
    _simulationFacade->pauseSimulation();
    printOverlayMessage("Pause");
}

void _MainWindow::onExit()
{
    _exitDialog->open();
}

void _MainWindow::finishFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(WindowController::get().getWindowData().window);
}

void _MainWindow::pushGlobalStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, Const::SliderBarWidth);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, Const::WindowsRounding);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImVec4)Const::HeaderHoveredColor);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, (ImVec4)Const::HeaderActiveColor);
    ImGui::PushStyleColor(ImGuiCol_Header, (ImVec4)Const::HeaderColor);
}

void _MainWindow::popGlobalStyle()
{
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
}
