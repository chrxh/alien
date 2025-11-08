#include "MainWindow.h"

#include <iostream>

#include <glad/glad.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Exceptions.h>
#include <Base/Resources.h>

#include <Network/NetworkService.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/SerializerService.h>

#include "AboutDialog.h"
#include "ActivateUserDialog.h"
#include "AlienGui.h"
#include "AutosaveWindow.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "CreatorWindow.h"
#include "DelayedExecutionController.h"
#include "DeleteUserDialog.h"
#include "DisplaySettingsDialog.h"
#include "EditSimulationDialog.h"
#include "EditorController.h"
#include "ExitDialog.h"
#include "FileTransferController.h"
#include "FpsController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "GettingStartedWindow.h"
#include "GpuSettingsDialog.h"
#include "GuiLogger.h"
#include "ImFileDialog.h"
#include "ImageToPatternDialog.h"
#include "LocationController.h"
#include "LogWindow.h"
#include "LoginController.h"
#include "LoginDialog.h"
#include "MainLoopController.h"
#include "MainLoopEntityController.h"
#include "MassOperationsDialog.h"
#include "MultiplierWindow.h"
#include "NetworkSettingsDialog.h"
#include "NetworkTransferController.h"
#include "NewPasswordDialog.h"
#include "NewSimulationDialog.h"
#include "OverlayController.h"
#include "PatternEditorWindow.h"
#include <EngineInterface/SimulationFacade.h>
#include <PersisterInterface/PersisterFacade.h>
#include "ResetPasswordDialog.h"
#include "SelectionWindow.h"
#include "SimulationInteractionController.h"
#include "SimulationParametersMainWindow.h"
#include "SimulationView.h"
#include "SpatialControlWindow.h"
#include "StatisticsWindow.h"
#include "StyleRepository.h"
#include "TemporalControlWindow.h"
#include "UiController.h"
#include "UploadSimulationDialog.h"
#include "Viewport.h"
#include "WindowController.h"
#include "implot.h"
#include "StartupCheckService.h"

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

_MainWindow::_MainWindow()
{
    IMGUI_CHECKVERSION();

    LogWindow::get().setup();

    StartupCheckService::get().check();

    log(Priority::Important, "initialize GLFW and OpenGL");
    initGlfwAndOpenGL();

    log(Priority::Important, "initialize GLAD");
    initGlad();

    log(Priority::Important, "initialize services");
    StyleRepository::get().setup();
    NetworkService::get().setup();

    log(Priority::Important, "initialize facades");
    _PersisterFacade::get()->setup();

    log(Priority::Important, "initialize main loop elements");
    Viewport::get().setup();
    EditorController::get().setup();
    SimulationView::get().setup();
    SimulationInteractionController::get().setup();
    StatisticsWindow::get().setup();
    TemporalControlWindow::get().setup();
    SpatialControlWindow::get().setup();
    SimulationParametersMainWindow::get().setup();
    LocationController::get().setup();
    GpuSettingsDialog::get().setup();
    MainLoopController::get().setup();
    ExitDialog::get().setup();
    MassOperationsDialog::get().setup();
    GettingStartedWindow::get().setup();
    NewSimulationDialog::get().setup();
    BrowserWindow::get().setup();
    ActivateUserDialog::get().setup();
    NewPasswordDialog::get().setup();
    LoginDialog::get().setup();
    UploadSimulationDialog::get().setup();
    ImageToPatternDialog::get().setup();
    AutosaveWindow::get().setup();
    OverlayController::get().setup();
    FileTransferController::get().setup();
    NetworkTransferController::get().setup();
    LoginController::get().setup();
    AboutDialog::get().setup();
    CreateUserDialog::get().setup();
    DeleteUserDialog::get().setup();
    DisplaySettingsDialog::get().setup();
    NetworkSettingsDialog::get().setup();
    NewPasswordDialog::get().setup();
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
    while (!MainLoopController::get().shouldClose()) {
        MainLoopController::get().process();
    }
}

void _MainWindow::shutdown()
{
    MainLoopController::get().shutdown();
    MainLoopEntityController::get().shutdown();
    SimulationView::get().shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(WindowController::get().getWindowData().window);
    glfwTerminate();

    _PersisterFacade::get()->shutdown();
    _SimulationFacade::get()->closeSimulation();

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
    //glfwSwapInterval(1);  //enable vsync
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
