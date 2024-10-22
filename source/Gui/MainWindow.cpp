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
#include "GenericMessageDialog.h"
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
#include "GenericFileDialog.h"
#include "ShaderWindow.h"
#include "GenomeEditorWindow.h"
#include "RadiationSourcesWindow.h"
#include "OverlayMessageController.h"
#include "ExitDialog.h"
#include "AutosaveWindow.h"
#include "FileTransferController.h"
#include "LoginController.h"
#include "NetworkTransferController.h"
#include "MainLoopEntityController.h"

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

    log(Priority::Important, "initialize services");
    StyleRepository::get().init();
    NetworkService::get().init();

    log(Priority::Important, "initialize facades");
    _persisterFacade->init(_simulationFacade);

    log(Priority::Important, "initialize windows");
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
    GpuSettingsDialog::get().init(_simulationFacade);
    StartupController::get().init(_simulationFacade, _persisterFacade);
    ExitDialog::get().init(_onExit);
    MassOperationsDialog::get().init(_simulationFacade);
    LogWindow::get().init(_logger);
    GettingStartedWindow::get().init();
    NewSimulationDialog::get().init(_simulationFacade);
    DisplaySettingsDialog::get().init();
    PatternAnalysisDialog::get().init(_simulationFacade);
    BrowserWindow::get().init(_simulationFacade, _persisterFacade);
    ActivateUserDialog::get().init(_simulationFacade);
    NewPasswordDialog::get().init(_simulationFacade);
    LoginDialog::get().init(_simulationFacade, _persisterFacade);
    UploadSimulationDialog::get().init(_simulationFacade);
    ImageToPatternDialog::get().init(_simulationFacade);
    AutosaveWindow::get().init(_simulationFacade, _persisterFacade);
    OverlayMessageController::get().init(_persisterFacade);
    FileTransferController::get().init(_persisterFacade, _simulationFacade);
    NetworkTransferController::get().init(_simulationFacade, _persisterFacade);
    LoginController::get().init(_simulationFacade, _persisterFacade);
    ShaderWindow::get().init();
    AboutDialog::get().init();
    ActivateUserDialog::get().init(_simulationFacade);
    CreateUserDialog::get().init();
    DeleteUserDialog::get().init();
    DisplaySettingsDialog::get().init(); 
    NetworkSettingsDialog::get().init();
    NewPasswordDialog::get().init(_simulationFacade);
    ResetPasswordDialog::get().init();
    GenericMessageDialog::get().init();
    GenericFileDialog::get().init();
    DelayedExecutionController::get().init();
    UiController::get().init();

    log(Priority::Important, "initialize file dialogs");
    initFileDialogs();

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

        switch (StartupController::get().getState()) {
        case StartupController::State::StartLoadSimulation:
            mainLoopForLoadingScreen();
            break;
        case StartupController::State::LoadingSimulation:
            mainLoopForLoadingScreen();
            break;
        case StartupController::State::FadeOutLoadingScreen:
            mainLoopForFadeoutLoadingScreen();
            break;
        case StartupController::State::FadeInUI:
            mainLoopForFadeInUI();
            break;
        case StartupController::State::Ready:
            mainLoopForUI();
            break;
        default:
            THROW_NOT_IMPLEMENTED();
        }
    }
}

void _MainWindow::shutdown()
{
    MainLoopEntityController::get().shutdown();

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

void _MainWindow::initFileDialogs()
{
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
}

void _MainWindow::mainLoopForLoadingScreen()
{
    StartupController::get().process();
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

void _MainWindow::mainLoopForFadeoutLoadingScreen()
{
    StartupController::get().process();
    renderSimulation();

    finishFrame();
}

void _MainWindow::mainLoopForFadeInUI()
{
    renderSimulation();

    pushGlobalStyle();

    processMenubar();
    MainLoopEntityController::get().process();
    OverlayMessageController::get().process();

    SimulationView::get().processControls(_renderSimulation);
    StartupController::get().process();

    popGlobalStyle();

    FpsController::get().processForceFps(WindowController::get().getFps());

    finishFrame();
}

void _MainWindow::mainLoopForUI()
{
    renderSimulation();

    pushGlobalStyle();

    processMenubar();
    MainLoopEntityController::get().process();
    OverlayMessageController::get().process();

    SimulationView::get().processControls(_renderSimulation);

    popGlobalStyle();

    FpsController::get().processForceFps(WindowController::get().getFps());

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
    if (ImGui::BeginMainMenuBar()) {
        if (AlienImGui::ShutdownButton()) {
            onExit();
        }
        ImGui::Dummy(ImVec2(10.0f, 0.0f));
        if (AlienImGui::BeginMenuButton(" " ICON_FA_GAMEPAD "  Simulation ", _simulationMenuToggled, "Simulation")) {
            if (ImGui::MenuItem("New", "CTRL+N")) {
                NewSimulationDialog::get().open();
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
                DeleteUserDialog::get().open();
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
            if (ImGui::MenuItem("Shader parameters", "ALT+6", ShaderWindow::get().isOn())) {
                ShaderWindow::get().setOn(!ShaderWindow::get().isOn());
            }
            if (ImGui::MenuItem("Autosave", "ALT+7", AutosaveWindow::get().isOn())) {
                AutosaveWindow::get().setOn(!AutosaveWindow::get().isOn());
            }
            if (ImGui::MenuItem("Log", "ALT+8", LogWindow::get().isOn())) {
                LogWindow::get().setOn(!LogWindow::get().isOn());
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_PEN_ALT "  Editor ", _editorMenuToggled, "Editor")) {
            if (ImGui::MenuItem("Activate", "ALT+E", SimulationInteractionController::get().isEditMode())) {
                SimulationInteractionController::get().setEditMode(!SimulationInteractionController::get().isEditMode());
            }
            ImGui::Separator();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode());
            if (ImGui::MenuItem("Selection", "ALT+S", SelectionWindow::get().isOn())) {
                SelectionWindow::get().setOn(!SelectionWindow::get().isOn());
            }
            if (ImGui::MenuItem("Creator", "ALT+R", CreatorWindow::get().isOn())) {
                CreatorWindow::get().setOn(!CreatorWindow::get().isOn());
            }
            if (ImGui::MenuItem("Pattern editor", "ALT+M", PatternEditorWindow::get().isOn())) {
                PatternEditorWindow::get().setOn(!PatternEditorWindow::get().isOn());
            }
            if (ImGui::MenuItem("Genome editor", "ALT+B", GenomeEditorWindow::get().isOn())) {
                GenomeEditorWindow::get().setOn(!GenomeEditorWindow::get().isOn());
            }
            if (ImGui::MenuItem("Multiplier", "ALT+A", MultiplierWindow::get().isOn())) {
                MultiplierWindow::get().setOn(!MultiplierWindow::get().isOn());
            }
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !PatternEditorWindow::get().isObjectInspectionPossible());
            if (ImGui::MenuItem("Inspect objects", "ALT+N")) {
                EditorController::get().onInspectSelectedObjects();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!SimulationInteractionController::get().isEditMode() || !PatternEditorWindow::get().isGenomeInspectionPossible());
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
                MassOperationsDialog::get().open();
                _toolsMenuToggled = false;
            }
            if (ImGui::MenuItem("Pattern analysis", "ALT+P")) {
                PatternAnalysisDialog::get().show();
                _toolsMenuToggled = false;
            }
            if (ImGui::MenuItem("Image converter", "ALT+G")) {
                ImageToPatternDialog::get().show();
                _toolsMenuToggled = false;
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_COG "  Settings ", _settingsMenuToggled, "Settings", false)) {
            if (ImGui::MenuItem("Auto save", "", AutosaveController::get().isOn())) {
                AutosaveController::get().setOn(!AutosaveController::get().isOn());
            }
            if (ImGui::MenuItem("CUDA settings", "ALT+C")) {
                GpuSettingsDialog::get().open();
            }
            if (ImGui::MenuItem("Display settings", "ALT+V")) {
                DisplaySettingsDialog::get().open();
            }
            if (ImGui::MenuItem("Network settings", "ALT+K")) {
                NetworkSettingsDialog::get().open();
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_LIFE_RING "  Help ", _helpMenuToggled, "Help")) {
            if (ImGui::MenuItem("About", "")) {
                AboutDialog::get().open();
                _helpMenuToggled = false;
            }
            if (ImGui::MenuItem("Getting started", "", GettingStartedWindow::get().isOn())) {
                GettingStartedWindow::get().setOn(!GettingStartedWindow::get().isOn());
            }
            AlienImGui::EndMenuButton();
        }
        ImGui::EndMainMenuBar();
    }

    //hotkeys
    auto& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_N)) {
            NewSimulationDialog::get().open();
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
            DeleteUserDialog::get().open();
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
            ShaderWindow::get().setOn(!ShaderWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_7)) {
            AutosaveWindow::get().setOn(!AutosaveWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_8)) {
            LogWindow::get().setOn(!LogWindow::get().isOn());
        }

        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_E)) {
            SimulationInteractionController::get().setEditMode(!SimulationInteractionController::get().isEditMode());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_S)) {
            SelectionWindow::get().setOn(!SelectionWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_M)) {
            PatternEditorWindow::get().setOn(!PatternEditorWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_B)) {
            GenomeEditorWindow::get().setOn(!GenomeEditorWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_R)) {
            CreatorWindow::get().setOn(!CreatorWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_A)) {
            MultiplierWindow::get().setOn(!MultiplierWindow::get().isOn());
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_N) && PatternEditorWindow::get().isObjectInspectionPossible()) {
            EditorController::get().onInspectSelectedObjects();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_F) && PatternEditorWindow::get().isGenomeInspectionPossible()) {
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
            GpuSettingsDialog::get().open();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_V)) {
            DisplaySettingsDialog::get().open();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F7)) {
            if (WindowController::get().isDesktopMode()) {
                WindowController::get().setWindowedMode();
            } else {
                WindowController::get().setDesktopMode();
            }
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_K)) {
            NetworkSettingsDialog::get().open();
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
            MassOperationsDialog::get().open();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_P)) {
            PatternAnalysisDialog::get().show();
        }
        if (io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_G)) {
            ImageToPatternDialog::get().show();
        }
    }
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
    ExitDialog::get().open();
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
