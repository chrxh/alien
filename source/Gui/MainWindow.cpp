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
#include "MainLoopController.h"
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
#include "ExitDialog.h"
#include "AutosaveWindow.h"
#include "FileTransferController.h"
#include "LoginController.h"
#include "NetworkTransferController.h"
#include "MainLoopEntityController.h"
#include "OverlayController.h"

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

    log(Priority::Important, "initialize main loop elements");
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
    MainLoopController::get().setup(_simulationFacade, _persisterFacade);
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
    OverlayController::get().setup(_persisterFacade);
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

    log(Priority::Important, "user interface initialized");
}

void _MainWindow::mainLoop()
{
    auto window = WindowController::get().getWindowData().window;
    while (!glfwWindowShouldClose(window) && !_onExit)
    {
        MainLoopController::get().process();
    }
}

void _MainWindow::shutdown()
{
    MainLoopEntityController::get().shutdown();
    SimulationView::get().shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(WindowController::get().getWindowData().window);
    glfwTerminate();

    _persisterFacade->shutdown();
    _simulationFacade->closeSimulation();

    NetworkService::get().shutdown();
}

void _MainWindow::initGlfwAndOpenGL()
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize Glfw.");
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glslVersion = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glslVersion = "#version 150";
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

    WindowController::get().setup();
    auto windowData = WindowController::get().getWindowData();
    glfwSetFramebufferSizeCallback(windowData.window, framebufferSizeCallback);
    glfwSwapInterval(1);  //enable vsync
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(windowData.window, true);  //setup Platform/Renderer back-ends
    ImGui_ImplOpenGL3_Init(glslVersion);
}

void _MainWindow::initGlad()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD.");
    }
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
