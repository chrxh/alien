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

    void framebufferSizeCallback(GLFWwindow* window, int width, int height)
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
    StyleRepository::get().setup();
    NetworkService::get().setup();

    log(Priority::Important, "initialize facades");
    _persisterFacade->setup(_simulationFacade);

    log(Priority::Important, "initialize gui elements");
    Viewport::get().setup(_simulationFacade);
    AutosaveController::get().setup(_simulationFacade);
    EditorController::get().setup(_simulationFacade);
    SimulationView::get().setup(_simulationFacade);
    SimulationInteractionController::get().setup(_simulationFacade);
    StatisticsWindow::get().setup(_simulationFacade);
    TemporalControlWindow::get().setup(_simulationFacade);
    SpatialControlWindow::get().setup(_simulationFacade);
    RadiationSourcesWindow::get().setup(_simulationFacade);
    SimulationParametersWindow::get().setup(_simulationFacade);
    GpuSettingsDialog::get().setup(_simulationFacade);
    StartupController::get().setup(_simulationFacade, _persisterFacade);
    ExitDialog::get().setup(_onExit);
    MassOperationsDialog::get().setup(_simulationFacade);
    LogWindow::get().setup(_logger);
    GettingStartedWindow::get().setup();
    NewSimulationDialog::get().setup(_simulationFacade);
    PatternAnalysisDialog::get().setup(_simulationFacade);
    BrowserWindow::get().setup(_simulationFacade, _persisterFacade);
    ActivateUserDialog::get().setup(_simulationFacade);
    NewPasswordDialog::get().setup(_simulationFacade);
    LoginDialog::get().setup(_simulationFacade, _persisterFacade);
    UploadSimulationDialog::get().setup(_simulationFacade);
    ImageToPatternDialog::get().setup(_simulationFacade);
    AutosaveWindow::get().setup(_simulationFacade, _persisterFacade);
    OverlayMessageController::get().setup(_persisterFacade);
    FileTransferController::get().setup(_persisterFacade, _simulationFacade);
    NetworkTransferController::get().setup(_simulationFacade, _persisterFacade);
    LoginController::get().setup(_simulationFacade, _persisterFacade);
    ShaderWindow::get().setup();
    AboutDialog::get().setup();
    ActivateUserDialog::get().setup(_simulationFacade);
    CreateUserDialog::get().setup();
    DeleteUserDialog::get().setup();
    DisplaySettingsDialog::get().setup(); 
    NetworkSettingsDialog::get().setup();
    NewPasswordDialog::get().setup(_simulationFacade);
    ResetPasswordDialog::get().setup();
    GenericMessageDialog::get().setup();
    GenericFileDialog::get().setup();
    DelayedExecutionController::get().setup();
    UiController::get().setup();

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

        int display_w, display_h;
        glfwGetFramebufferSize(WindowController::get().getWindowData().window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

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

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(WindowController::get().getWindowData().window);
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
    WindowController::get().setup();
    auto windowData = WindowController::get().getWindowData();
    glfwSetFramebufferSizeCallback(windowData.window, framebufferSizeCallback);
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

    //background color
    glClearColor(0, 0, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void _MainWindow::mainLoopForFadeoutLoadingScreen()
{
    StartupController::get().process();
    SimulationView::get().draw(_renderSimulation);
}

void _MainWindow::mainLoopForFadeInUI()
{
    SimulationView::get().draw(_renderSimulation);

    pushGlobalStyle();

    processMenubar();
    MainLoopEntityController::get().process();
    OverlayMessageController::get().process();

    SimulationView::get().processControls(_renderSimulation);
    StartupController::get().process();

    popGlobalStyle();

    FpsController::get().processForceFps(WindowController::get().getFps());
}

void _MainWindow::mainLoopForUI()
{
    SimulationView::get().draw(_renderSimulation);

    pushGlobalStyle();

    processMenubar();
    MainLoopEntityController::get().process();
    OverlayMessageController::get().process();

    SimulationView::get().processControls(_renderSimulation);

    popGlobalStyle();

    FpsController::get().processForceFps(WindowController::get().getFps());
}

void _MainWindow::processMenubar()
{
    auto& io = ImGui::GetIO();

    AlienImGui::BeginMenuBar();
    AlienImGui::MenuShutdownButton([&] { onExit(); });
    ImGui::Dummy(ImVec2(scale(10.0f), 0.0f));

    AlienImGui::BeginMenu(" " ICON_FA_GAMEPAD "  Simulation ", _simulationMenuToggled);
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("New").keyCtrl(true).key(ImGuiKey_N), [&] { NewSimulationDialog::get().open(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Open").keyCtrl(true).key(ImGuiKey_O), [&] { FileTransferController::get().onOpenSimulation(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Save").keyCtrl(true).key(ImGuiKey_S), [&] { FileTransferController::get().onSaveSimulation(); });
    AlienImGui::MenuSeparator();
    auto running = _simulationFacade->isSimulationRunning();
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Run").key(ImGuiKey_Space).disabled(running).closeMenuWhenItemClicked(false), [&] { onRunSimulation(); });
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("Pause").key(ImGuiKey_Space).disabled(!running).closeMenuWhenItemClicked(false), [&] {
        onPauseSimulation();
    });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_GLOBE "  Network ", _networkMenuToggled);
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Browser").keyAlt(true).key(ImGuiKey_W).closeMenuWhenItemClicked(false).selected(BrowserWindow::get().isOn()),
        [&] { BrowserWindow::get().setOn(!BrowserWindow::get().isOn()); });
    AlienImGui::MenuSeparator();
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Login")
            .keyAlt(true)
            .key(ImGuiKey_L)
            .disabled(NetworkService::get().isLoggedIn()),
        [&] { LoginDialog::get().open(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Logout")
            .keyAlt(true)
            .key(ImGuiKey_T)
            .closeMenuWhenItemClicked(false)
            .disabled(!NetworkService::get().isLoggedIn()),
        [&] {
            NetworkService::get().logout();
            BrowserWindow::get().onRefresh();
        });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Upload simulation").keyAlt(true).key(ImGuiKey_D).disabled(!NetworkService::get().isLoggedIn()),
        [&] { UploadSimulationDialog::get().open(NetworkResourceType_Simulation); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Upload genome").keyAlt(true).key(ImGuiKey_Q).disabled(!NetworkService::get().isLoggedIn()),
        [&] { UploadSimulationDialog::get().open(NetworkResourceType_Genome); });
    AlienImGui::MenuSeparator();
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Delete user").keyAlt(true).key(ImGuiKey_J).disabled(!NetworkService::get().isLoggedIn()),
        [&] { DeleteUserDialog::get().open(); });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_WINDOW_RESTORE "  Windows ", _windowMenuToggled);
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Temporal control")
            .keyAlt(true)
            .key(ImGuiKey_1)
            .selected(TemporalControlWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { TemporalControlWindow::get().setOn(!TemporalControlWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Spatial control")
            .keyAlt(true)
            .key(ImGuiKey_2)
            .selected(SpatialControlWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { SpatialControlWindow::get().setOn(!SpatialControlWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Statistics")
            .keyAlt(true)
            .key(ImGuiKey_3)
            .selected(StatisticsWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { StatisticsWindow::get().setOn(!StatisticsWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Simulation parameters")
            .keyAlt(true)
            .key(ImGuiKey_4)
            .selected(SimulationParametersWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { SimulationParametersWindow::get().setOn(!SimulationParametersWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Radiation sources")
            .keyAlt(true)
            .key(ImGuiKey_5)
            .selected(RadiationSourcesWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { RadiationSourcesWindow::get().setOn(!RadiationSourcesWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Shader parameters")
            .keyAlt(true)
            .key(ImGuiKey_6)
            .selected(ShaderWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { ShaderWindow::get().setOn(!ShaderWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Autosave")
            .keyAlt(true)
            .key(ImGuiKey_7)
            .selected(AutosaveWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { AutosaveWindow::get().setOn(!AutosaveWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Log")
            .keyAlt(true)
            .key(ImGuiKey_8).selected(LogWindow::get().isOn())
            .closeMenuWhenItemClicked(false),
        [&] { LogWindow::get().setOn(!LogWindow::get().isOn()); });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_PEN_ALT "  Editor ", _editorMenuToggled);
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Activate")
            .keyAlt(true)
            .key(ImGuiKey_E)
            .selected(SimulationInteractionController::get().isEditMode())
            .closeMenuWhenItemClicked(false),
        [&] { SimulationInteractionController::get().setEditMode(!SimulationInteractionController::get().isEditMode()); });
    AlienImGui::MenuSeparator();
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Selection")
            .keyAlt(true)
            .key(ImGuiKey_S)
            .selected(SelectionWindow::get().isOn())
            .disabled(!SimulationInteractionController::get().isEditMode())
            .closeMenuWhenItemClicked(false),
        [&] { SelectionWindow::get().setOn(!SelectionWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Creator")
            .keyAlt(true)
            .key(ImGuiKey_R)
            .selected(CreatorWindow::get().isOn())
            .disabled(!SimulationInteractionController::get().isEditMode())
            .closeMenuWhenItemClicked(false),
        [&] { CreatorWindow::get().setOn(!CreatorWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Pattern editor")
            .keyAlt(true)
            .key(ImGuiKey_M)
            .selected(PatternEditorWindow::get().isOn())
            .disabled(!SimulationInteractionController::get().isEditMode())
            .closeMenuWhenItemClicked(false),
        [&] { PatternEditorWindow::get().setOn(!PatternEditorWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Genome editor")
            .keyAlt(true)
            .key(ImGuiKey_B)
            .selected(GenomeEditorWindow::get().isOn())
            .disabled(!SimulationInteractionController::get().isEditMode())
            .closeMenuWhenItemClicked(false),
        [&] { GenomeEditorWindow::get().setOn(!GenomeEditorWindow::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Multiplier")
            .keyAlt(true)
            .key(ImGuiKey_A)
            .selected(MultiplierWindow::get().isOn())
            .disabled(!SimulationInteractionController::get().isEditMode())
            .closeMenuWhenItemClicked(false),
        [&] { MultiplierWindow::get().setOn(!MultiplierWindow::get().isOn()); });
    AlienImGui::MenuSeparator();
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Inspect objects")
            .keyAlt(true)
            .key(ImGuiKey_N)
            .disabled(!SimulationInteractionController::get().isEditMode() || !PatternEditorWindow::get().isObjectInspectionPossible()),
        [&] { EditorController::get().onInspectSelectedObjects(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Inspect principal genome")
            .keyAlt(true)
            .key(ImGuiKey_F)
            .disabled(!SimulationInteractionController::get().isEditMode() || !PatternEditorWindow::get().isGenomeInspectionPossible()),
        [&] { EditorController::get().onInspectSelectedGenomes(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Close inspections")
            .key(ImGuiKey_Escape)
            .disabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().areInspectionWindowsActive()),
        [&] { EditorController::get().onCloseAllInspectorWindows(); });
    AlienImGui::MenuSeparator();
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Copy")
            .keyCtrl(true)
            .key(ImGuiKey_C)
            .disabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isCopyingPossible()),
        [&] { EditorController::get().onCopy(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Paste")
            .keyCtrl(true)
            .key(ImGuiKey_V)
            .disabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isPastingPossible()),
        [&] { EditorController::get().onPaste(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Delete")
            .key(ImGuiKey_Delete)
            .disabled(!SimulationInteractionController::get().isEditMode() || !EditorController::get().isCopyingPossible()),
        [&] { EditorController::get().onDelete(); });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_EYE "  View ", _viewMenuToggled);
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters()
            .name("Information overlay")
            .keyAlt(true)
            .key(ImGuiKey_O)
            .selected(SimulationView::get().isOverlayActive())
            .closeMenuWhenItemClicked(false),
        [&] { SimulationView::get().setOverlayActive(!SimulationView::get().isOverlayActive()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Render UI").keyAlt(true).key(ImGuiKey_U).selected(UiController::get().isOn()).closeMenuWhenItemClicked(false),
        [&] { UiController::get().setOn(!UiController::get().isOn()); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Render simulation").keyAlt(true).key(ImGuiKey_I).selected(_renderSimulation).closeMenuWhenItemClicked(false),
        [&] { _renderSimulation = !_renderSimulation; });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_TOOLS "  Tools ", _toolsMenuToggled);
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Mass operations").keyAlt(true).key(ImGuiKey_H), [&] { MassOperationsDialog::get().open(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Pattern analysis").keyAlt(true).key(ImGuiKey_P), [&] { PatternAnalysisDialog::get().show(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Image converter").keyAlt(true).key(ImGuiKey_G), [&] { ImageToPatternDialog::get().show(); });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_COG "  Settings ", _settingsMenuToggled, false);
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("Auto save").selected(AutosaveController::get().isOn()).closeMenuWhenItemClicked(false), [&] {
        AutosaveController::get().setOn(!AutosaveController::get().isOn());
    });
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("CUDA settings").keyAlt(true).key(ImGuiKey_C), [&] { GpuSettingsDialog::get().open(); });
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("Display settings").keyAlt(true).key(ImGuiKey_V), [&] { DisplaySettingsDialog::get().open(); });
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("Network settings").keyAlt(true).key(ImGuiKey_K), [&] { NetworkSettingsDialog::get().open(); });
    AlienImGui::EndMenu();

    AlienImGui::BeginMenu(" " ICON_FA_LIFE_RING "  Help ", _helpMenuToggled);
    AlienImGui::MenuItem(AlienImGui::MenuItemParameters().name("About"), [&] { AboutDialog::get().open(); });
    AlienImGui::MenuItem(
        AlienImGui::MenuItemParameters().name("Getting started").selected(GettingStartedWindow::get().isOn()).closeMenuWhenItemClicked(false),
        [&] { GettingStartedWindow::get().setOn(!GettingStartedWindow::get().isOn()); });
    AlienImGui::EndMenu();
    AlienImGui::EndMenuBar();

    //further hotkeys
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_F7)) {
            if (WindowController::get().isDesktopMode()) {
                WindowController::get().setWindowedMode();
            } else {
                WindowController::get().setDesktopMode();
            }
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
