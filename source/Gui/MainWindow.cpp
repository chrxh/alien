#include "MainWindow.h"

#include <iostream>

#include <boost/algorithm/string.hpp>

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

#include "PersisterInterface/PersisterController.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"
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

namespace
{
    void glfwErrorCallback(int error, const char* description)
    {
        throw std::runtime_error("Glfw error " + std::to_string(error) + ": " + description);
    }

    _SimulationView* simulationViewPtr;
    void framebuffer_size_callback(GLFWwindow* window, int width, int height)
    {
        if (width > 0 && height > 0) {
            simulationViewPtr->resize({width, height});
            glViewport(0, 0, width, height);
        }
    }
}

_MainWindow::_MainWindow(SimulationController const& simController, PersisterController const& persisterController, GuiLogger const& logger)
    : _logger(logger)
    , _simController(simController)
    , _persisterController(persisterController)
{
    IMGUI_CHECKVERSION();


    log(Priority::Important, "initialize GLFW and OpenGL");
    auto glfwVersion = initGlfw();
    WindowController::init();
    auto windowData = WindowController::getWindowData();
    glfwSetFramebufferSizeCallback(windowData.window, framebuffer_size_callback);
    glfwSwapInterval(1);  //enable vsync
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(windowData.window, true);  //setup Platform/Renderer back-ends
    ImGui_ImplOpenGL3_Init(glfwVersion);

    log(Priority::Important, "initialize GLAD");
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    //init services
    StyleRepository::getInstance().init();
    NetworkService::init();

    //init controllers, windows and dialogs
    Viewport::init(_simController);
    _uiController = std::make_shared<_UiController>();
    _autosaveController = std::make_shared<_AutosaveController>(_simController);
    _editorController =
        std::make_shared<_EditorController>(_simController);
    _simulationView = std::make_shared<_SimulationView>(_simController);
    _simInteractionController = std::make_shared<_SimulationInteractionController>(_simController, _editorController, _simulationView);
    simulationViewPtr = _simulationView.get();
    _statisticsWindow = std::make_shared<_StatisticsWindow>(_simController);
    _temporalControlWindow = std::make_shared<_TemporalControlWindow>(_simController, _statisticsWindow);
    _spatialControlWindow = std::make_shared<_SpatialControlWindow>(_simController, _temporalControlWindow);
    _radiationSourcesWindow = std::make_shared<_RadiationSourcesWindow>(_simController, _simInteractionController);
    _simulationParametersWindow = std::make_shared<_SimulationParametersWindow>(_simController, _radiationSourcesWindow, _simInteractionController);
    _gpuSettingsDialog = std::make_shared<_GpuSettingsDialog>(_simController);
    _startupController = std::make_shared<_StartupController>(_simController, _temporalControlWindow);
    _exitDialog = std::make_shared<_ExitDialog>(_onExit);
    _aboutDialog = std::make_shared<_AboutDialog>();
    _massOperationsDialog = std::make_shared<_MassOperationsDialog>(_simController);
    _logWindow = std::make_shared<_LogWindow>(_logger);
    _gettingStartedWindow = std::make_shared<_GettingStartedWindow>();
    _newSimulationDialog = std::make_shared<_NewSimulationDialog>(_simController, _temporalControlWindow, _statisticsWindow);
    _displaySettingsDialog = std::make_shared<_DisplaySettingsDialog>();
    _patternAnalysisDialog = std::make_shared<_PatternAnalysisDialog>(_simController);
    _fpsController = std::make_shared<_FpsController>();
    _browserWindow =
        std::make_shared<_BrowserWindow>(_simController, _statisticsWindow, _temporalControlWindow, _editorController);
    _activateUserDialog = std::make_shared<_ActivateUserDialog>(_simController, _browserWindow);
    _createUserDialog = std::make_shared<_CreateUserDialog>(_activateUserDialog);
    _newPasswordDialog = std::make_shared<_NewPasswordDialog>(_simController, _browserWindow);
    _resetPasswordDialog = std::make_shared<_ResetPasswordDialog>(_newPasswordDialog);
    _loginDialog = std::make_shared<_LoginDialog>(_simController, _browserWindow, _createUserDialog, _activateUserDialog, _resetPasswordDialog);
    _uploadSimulationDialog = std::make_shared<_UploadSimulationDialog>(
        _browserWindow, _loginDialog, _simController, _editorController->getGenomeEditorWindow());
    _editSimulationDialog = std::make_shared<_EditSimulationDialog>(_browserWindow);
    _deleteUserDialog = std::make_shared<_DeleteUserDialog>(_browserWindow);
    _networkSettingsDialog = std::make_shared<_NetworkSettingsDialog>(_browserWindow);
    _imageToPatternDialog = std::make_shared<_ImageToPatternDialog>(_simController);
    _shaderWindow = std::make_shared<_ShaderWindow>(_simulationView);
    _autosaveWindow = std::make_shared<_AutosaveWindow>(_simController, _persisterController);
    _persisterController->init(_simController);
    OverlayMessageController::getInstance().init(_persisterController);

    //cyclic references
    _browserWindow->registerCyclicReferences(_loginDialog, _uploadSimulationDialog, _editSimulationDialog, _editorController->getGenomeEditorWindow());
    _activateUserDialog->registerCyclicReferences(_createUserDialog);
    _editorController->registerCyclicReferences(_uploadSimulationDialog, _simInteractionController);

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

    _window = windowData.window;

    log(Priority::Important, "main window initialized");
}

void _MainWindow::mainLoop()
{
    while (!glfwWindowShouldClose(_window) && !_onExit)
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

     //   ImGui::ShowDemoWindow(NULL);

        switch (_startupController->getState()) {
        case _StartupController::State::Unintialized:
            processUninitialized();
            break;
        case _StartupController::State::LoadSimulation:
            processRequestLoading();
            break;
        case _StartupController::State::FadeOutLoadingScreen:
            processLoadingSimulation();
            break;
        case _StartupController::State::LoadingControls:
            processLoadingControls();
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
    WindowController::shutdown();
    _autosaveController->shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(_window);
    glfwTerminate();

    _simulationView.reset();

    _persisterController->shutdown();
    _simController->closeSimulation();
    NetworkService::shutdown();
}

char const* _MainWindow::initGlfw()
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
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    return glsl_version;
}

void _MainWindow::processUninitialized()
{
    _startupController->process();

    // render mainData
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(_window);
}

void _MainWindow::processRequestLoading()
{
    _startupController->process();
    renderSimulation();
}

void _MainWindow::processLoadingSimulation()
{
    _startupController->process();
    renderSimulation();
}

void _MainWindow::processLoadingControls()
{
    pushGlobalStyle();

    processMenubar();
    processDialogs();
    processWindows();
    processControllers();

    _uiController->process();
    _simulationView->processControls(_renderSimulation);
    _startupController->process();

    popGlobalStyle();

    renderSimulation();
}

void _MainWindow::processReady()
{
    pushGlobalStyle();

    processMenubar();
    processDialogs();
    processWindows();
    processControllers();
    _uiController->process();
    _simulationView->processControls(_renderSimulation);

    popGlobalStyle();

    renderSimulation();
}

void _MainWindow::renderSimulation()
{
    int display_w, display_h;
    glfwGetFramebufferSize(_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    _simulationView->draw(_renderSimulation);
    ImGui::Render();

    _fpsController->processForceFps(WindowController::getFps());

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(_window);
}

void _MainWindow::processMenubar()
{
    auto selectionWindow = _editorController->getSelectionWindow();
    auto patternEditorWindow = _editorController->getPatternEditorWindow();
    auto creatorWindow = _editorController->getCreatorWindow();
    auto multiplierWindow = _editorController->getMultiplierWindow();
    auto genomeEditorWindow = _editorController->getGenomeEditorWindow();

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
                onOpenSimulation();
                _simulationMenuToggled = false;
            }
            if (ImGui::MenuItem("Save", "CTRL+S")) {
                onSaveSimulation();
                _simulationMenuToggled = false;
            }
            ImGui::Separator();
            ImGui::BeginDisabled(_simController->isSimulationRunning());
            if (ImGui::MenuItem("Run", "SPACE")) {
                onRunSimulation();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!_simController->isSimulationRunning());
            if (ImGui::MenuItem("Pause", "SPACE")) {
                onPauseSimulation();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_GLOBE "  Network ", _networkMenuToggled, "Network", false)) {
            if (ImGui::MenuItem("Browser", "ALT+W", _browserWindow->isOn())) {
                _browserWindow->setOn(!_browserWindow->isOn());
            }
            ImGui::Separator();
            ImGui::BeginDisabled((bool)NetworkService::getLoggedInUserName());
            if (ImGui::MenuItem("Login", "ALT+L")) {
                _loginDialog->open();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!NetworkService::getLoggedInUserName());
            if (ImGui::MenuItem("Logout", "ALT+T")) {
                NetworkService::logout();
                _browserWindow->onRefresh();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!NetworkService::getLoggedInUserName());
            if (ImGui::MenuItem("Upload simulation", "ALT+D")) {
                _uploadSimulationDialog->open(NetworkResourceType_Simulation);
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!NetworkService::getLoggedInUserName());
            if (ImGui::MenuItem("Upload genome", "ALT+Q")) {
                _uploadSimulationDialog->open(NetworkResourceType_Genome);
            }
            ImGui::EndDisabled();

            ImGui::Separator();
            ImGui::BeginDisabled(!NetworkService::getLoggedInUserName());
            if (ImGui::MenuItem("Delete user", "ALT+J")) {
                _deleteUserDialog->open();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_WINDOW_RESTORE "  Windows ", _windowMenuToggled, "Windows")) {
            if (ImGui::MenuItem("Temporal control", "ALT+1", _temporalControlWindow->isOn())) {
                _temporalControlWindow->setOn(!_temporalControlWindow->isOn());
            }
            if (ImGui::MenuItem("Spatial control", "ALT+2", _spatialControlWindow->isOn())) {
                _spatialControlWindow->setOn(!_spatialControlWindow->isOn());
            }
            if (ImGui::MenuItem("Statistics", "ALT+3", _statisticsWindow->isOn())) {
                _statisticsWindow->setOn(!_statisticsWindow->isOn());
            }
            if (ImGui::MenuItem("Simulation parameters", "ALT+4", _simulationParametersWindow->isOn())) {
                _simulationParametersWindow->setOn(!_simulationParametersWindow->isOn());
            }
            if (ImGui::MenuItem("Radiation sources", "ALT+5", _radiationSourcesWindow->isOn())) {
                _radiationSourcesWindow->setOn(!_radiationSourcesWindow->isOn());
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
            if (ImGui::MenuItem("Activate", "ALT+E", _simInteractionController->isEditMode())) {
                _simInteractionController->setEditMode(!_simInteractionController->isEditMode());
            }
            ImGui::Separator();
            ImGui::BeginDisabled(!_simInteractionController->isEditMode());
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
            ImGui::BeginDisabled(!_simInteractionController->isEditMode() || !_editorController->isObjectInspectionPossible());
            if (ImGui::MenuItem("Inspect objects", "ALT+N")) {
                _editorController->onInspectSelectedObjects();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!_simInteractionController->isEditMode() || !_editorController->isGenomeInspectionPossible());
            if (ImGui::MenuItem("Inspect principal genome", "ALT+F")) {
                _editorController->onInspectSelectedGenomes();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!_simInteractionController->isEditMode() || !_editorController->areInspectionWindowsActive());
            if (ImGui::MenuItem("Close inspections", "ESC")) {
                _editorController->onCloseAllInspectorWindows();
            }
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::BeginDisabled(!_simInteractionController->isEditMode() || !_editorController->isCopyingPossible());
            if (ImGui::MenuItem("Copy", "CTRL+C")) {
                _editorController->onCopy();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!_simInteractionController->isEditMode() || !_editorController->isPastingPossible());
            if (ImGui::MenuItem("Paste", "CTRL+V")) {
                _editorController->onPaste();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_EYE "  View ", _viewMenuToggled, "View")) {
            if (ImGui::MenuItem("Information overlay", "ALT+O", _simulationView->isOverlayActive())) {
                _simulationView->setOverlayActive(!_simulationView->isOverlayActive());
            }
            if (ImGui::MenuItem("Render UI", "ALT+U", _uiController->isOn())) {
                _uiController->setOn(!_uiController->isOn());
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
            if (ImGui::MenuItem("Auto save", "", _autosaveController->isOn())) {
                _autosaveController->setOn(!_autosaveController->isOn());
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
        if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_N)) {
            _newSimulationDialog->open();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_O)) {
            onOpenSimulation();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_S)) {
            onSaveSimulation();
        }
        if (ImGui::IsKeyPressed(GLFW_KEY_SPACE)) {
            if (_simController->isSimulationRunning()) {
                onPauseSimulation();
            } else {
                onRunSimulation();
            }
            
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_W)) {
            _browserWindow->setOn(!_browserWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_L) && !NetworkService::getLoggedInUserName()) {
            _loginDialog->open();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_T)) {
            NetworkService::logout();
            _browserWindow->onRefresh();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_D) && NetworkService::getLoggedInUserName()) {
            _uploadSimulationDialog->open(NetworkResourceType_Simulation);
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_Q) && NetworkService::getLoggedInUserName()) {
            _uploadSimulationDialog->open(NetworkResourceType_Genome);
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_J) && NetworkService::getLoggedInUserName()) {
            _deleteUserDialog->open();
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_1)) {
            _temporalControlWindow->setOn(!_temporalControlWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_2)) {
            _spatialControlWindow->setOn(!_spatialControlWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_3)) {
            _statisticsWindow->setOn(!_statisticsWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_4)) {
            _simulationParametersWindow->setOn(!_simulationParametersWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_5)) {
            _radiationSourcesWindow->setOn(!_radiationSourcesWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_6)) {
            _shaderWindow->setOn(!_shaderWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_7)) {
            _autosaveWindow->setOn(!_autosaveWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_8)) {
            _logWindow->setOn(!_logWindow->isOn());
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_E)) {
            _simInteractionController->setEditMode(!_simInteractionController->isEditMode());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_S)) {
            selectionWindow->setOn(!selectionWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_M)) {
            patternEditorWindow->setOn(!patternEditorWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_B)) {
            genomeEditorWindow->setOn(!genomeEditorWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_R)) {
            creatorWindow->setOn(!creatorWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_A)) {
            multiplierWindow->setOn(!multiplierWindow->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_N) && _editorController->isObjectInspectionPossible()) {
            _editorController->onInspectSelectedObjects();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_F) && _editorController->isGenomeInspectionPossible()) {
            _editorController->onInspectSelectedGenomes();
        }
        if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE)) {
            _editorController->onCloseAllInspectorWindows();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_C) && _editorController->isCopyingPossible()) {
            _editorController->onCopy();
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_V) && _editorController->isPastingPossible()) {
            _editorController->onPaste();
        }
        if (ImGui::IsKeyPressed(GLFW_KEY_DELETE) ) {
            _editorController->onDelete();
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_C)) {
            _gpuSettingsDialog->open();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_V)) {
            _displaySettingsDialog->open();
        }
        if (ImGui::IsKeyPressed(GLFW_KEY_F7)) {
            if (WindowController::isDesktopMode()) {
                WindowController::setWindowedMode();
            } else {
                WindowController::setDesktopMode();
            }
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_K)) {
            _networkSettingsDialog->open();
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_O)) {
            _simulationView->setOverlayActive(!_simulationView->isOverlayActive());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_U)) {
            _uiController->setOn(!_uiController->isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_I)) {
            _renderSimulation = !_renderSimulation;
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_H)) {
            _massOperationsDialog->show();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_P)) {
            _patternAnalysisDialog->show();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_G)) {
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
    _loginDialog->process();
    _createUserDialog->process();
    _activateUserDialog->process();
    _uploadSimulationDialog->process();
    _editSimulationDialog->process();
    _deleteUserDialog->process();
    _networkSettingsDialog->process();
    _resetPasswordDialog->process();
    _newPasswordDialog->process();
    _exitDialog->process();

    MessageDialog::getInstance().process();
    GenericFileDialogs::getInstance().process();
}

void _MainWindow::processWindows()
{
    _temporalControlWindow->process();
    _spatialControlWindow->process();
    _simInteractionController->process();
    _statisticsWindow->process();
    _simulationParametersWindow->process();
    _logWindow->process();
    _browserWindow->process();
    _gettingStartedWindow->process();
    _shaderWindow->process();
    _radiationSourcesWindow->process();
    _autosaveWindow->process();
}

void _MainWindow::processControllers()
{
    _autosaveController->process();
    _editorController->process();
    OverlayMessageController::getInstance().process();
    DelayedExecutionController::getInstance().process();
    auto criticalErrors = _persisterController->fetchCriticalErrorInfos();
    if (!criticalErrors.empty()) {
        std::vector<std::string> errorMessages;
        for (auto const& error : criticalErrors) {
            errorMessages.emplace_back(error.message);
        }
        MessageDialog::getInstance().information("Error", boost::join(errorMessages, "\n\n"));
    }
}

void _MainWindow::onOpenSimulation()
{
    GenericFileDialogs::getInstance().showOpenFileDialog(
        "Open simulation", "Simulation file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            printOverlayMessage("Loading ...");
            delayedExecution([firstFilename = firstFilename, this] {
                DeserializedSimulation deserializedData;
                if (SerializerService::deserializeSimulationFromFiles(deserializedData, firstFilename.string())) {
                    _simController->closeSimulation();

                    std::optional<std::string> errorMessage;
                    try {
                        _simController->newSimulation(
                            firstFilename.stem().string(),
                            deserializedData.auxiliaryData.timestep,
                            deserializedData.auxiliaryData.generalSettings,
                            deserializedData.auxiliaryData.simulationParameters);
                        _simController->setClusteredSimulationData(deserializedData.mainData);
                        _simController->setStatisticsHistory(deserializedData.statistics);
                        _simController->setRealTime(deserializedData.auxiliaryData.realTime);
                    } catch (CudaMemoryAllocationException const& exception) {
                        errorMessage = exception.what();
                    } catch (...) {
                        errorMessage = "Failed to load simulation.";
                    }

                    if (errorMessage) {
                        showMessage("Error", *errorMessage);
                        _simController->closeSimulation();
                        _simController->newSimulation(
                            std::nullopt,
                            deserializedData.auxiliaryData.timestep,
                            deserializedData.auxiliaryData.generalSettings,
                            deserializedData.auxiliaryData.simulationParameters);
                    }

                    Viewport::setCenterInWorldPos(deserializedData.auxiliaryData.center);
                    Viewport::setZoomFactor(deserializedData.auxiliaryData.zoom);
                    _temporalControlWindow->onSnapshot();
                    printOverlayMessage(firstFilename.filename().string());
                } else {
                    showMessage("Open simulation", "The selected file could not be opened.");
                }
            });
        });
}

void _MainWindow::onSaveSimulation()
{
    GenericFileDialogs::getInstance().showSaveFileDialog(
        "Save simulation", "Simulation file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();
            printOverlayMessage("Saving ...");
            _persisterController->scheduleSaveSimulationToDisc(firstFilename.string(), true, Viewport::getZoomFactor(), Viewport::getCenterInWorldPos());
        });
}

void _MainWindow::onRunSimulation()
{
    _simController->runSimulation();
    printOverlayMessage("Run");
}

void _MainWindow::onPauseSimulation()
{
    _simController->pauseSimulation();
    printOverlayMessage("Pause");
}

void _MainWindow::onExit()
{
    _exitDialog->open();
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
