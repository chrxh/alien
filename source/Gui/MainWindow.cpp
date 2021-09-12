#include "MainWindow.h"

#include <iostream>

#include <glad/glad.h>

#include "EngineInterface/Serializer.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineImpl/SimulationController.h"

#include "SimulationView.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "ImFileDialog.h"

namespace
{
    void glfwErrorCallback(int error, const char* description)
    {
        std::cerr << "Glfw Error " << error << ": " << description << std::endl;
    }
}

GLFWwindow* _MainWindow::init(SimulationController const& simController)
{
    _simController = simController;
    _simulationView = boost::make_shared<_SimulationView>();

    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        return nullptr;
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

    // Create window with graphics context
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor(); // The primary monitor.. Later Occulus?..
    auto mode = glfwGetVideoMode(primaryMonitor);
    auto screenWidth = mode->width;
    auto screenHeight = mode->height;

    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "alien", primaryMonitor, NULL);
    if (window == NULL) {
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

                         // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    //ImGui::StyleColorsDark();
//    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

/*
    glfwSetMouseButtonCallback(window, mouseClickEvent);
    glfwSetCursorPosCallback(window, mouseMoveEvent);
*/

    //    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return nullptr;
    }

    _simulationView->init(simController, {mode->width, mode->height}, 4);

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

        return (void*)tex;
    };
    ifd::FileDialog::Instance().DeleteTexture = [](void* tex) {
        GLuint texID = (GLuint)((uintptr_t)tex);
        glDeleteTextures(1, &texID);
    };
    return window;
}

void _MainWindow::mainLoop(GLFWwindow* window)
{
    bool show_demo_window = true;
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);
        {}

        drawToolbar();
        drawMenubar();
        drawDialogs();
        _simulationView->processControls();

        // render content
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        _simulationView->processContent();

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
}

void _MainWindow::shutdown(GLFWwindow* window)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    _simulationView.reset();
}

void _MainWindow::drawMenubar()
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Simulation")) {
            if (ImGui::MenuItem("Open", "CTRL+O")) {
                ifd::FileDialog::Instance().Open(
                    "SimulationOpenDialog",
                    "Open a simulation",
                    "Simulation file (*.sim){.sim},.*",
                    true);
            }
            if (ImGui::MenuItem("Close", "ALT+F4")) {
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void _MainWindow::drawToolbar()
{
/*
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + 19));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 50));

    ImGuiWindowFlags windowFlags = 0
        | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings
        ;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("TOOLBAR", NULL, windowFlags);
    ImGui::PopStyleVar();
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::SameLine();
    ImGui::Button("Zoom in", ImVec2(0, 37));
    ImGui::SameLine();
    ImGui::Button("Zoom out", ImVec2(0, 37));

    ImGui::End();
*/
}

void _MainWindow::drawDialogs()
{
    // Simple window
/*
    ImGui::Begin("Control Panel");
    if (ImGui::Button("Open directory"))
        ifd::FileDialog::Instance().Open("DirectoryOpenDialog", "Open a directory", "");
    if (ImGui::Button("Save file"))
        ifd::FileDialog::Instance().Save("ShaderSaveDialog", "Save a shader", "*.sprj {.sprj}");
    ImGui::End();
*/

    // file dialogs
    if (ifd::FileDialog::Instance().IsDone("SimulationOpenDialog")) {
        if (ifd::FileDialog::Instance().HasResult()) {
            const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
            auto firstFilename = res.front();

            _simController->closeSimulation();

            Serializer serializer = boost::make_shared<_Serializer>();
            SerializedSimulation serializedData;
            serializer->loadSimulationDataFromFile(firstFilename.string(), serializedData);

            auto deserializedData = serializer->deserializeSimulation(serializedData);

            _simController->newSimulation(
                deserializedData.generalSettings.worldSize,
                deserializedData.timestep,
                deserializedData.simulationParameters,
                deserializedData.generalSettings.gpuConstants);
            _simController->updateData(deserializedData.content);
            _simController->runSimulation();
        }
        ifd::FileDialog::Instance().Close();
    }
/*
    if (ifd::FileDialog::Instance().IsDone("DirectoryOpenDialog")) {
        if (ifd::FileDialog::Instance().HasResult()) {
            std::string res = ifd::FileDialog::Instance().GetResult().u8string();
            printf("DIRECTORY[%s]\n", res.c_str());
        }
        ifd::FileDialog::Instance().Close();
    }
    if (ifd::FileDialog::Instance().IsDone("ShaderSaveDialog")) {
        if (ifd::FileDialog::Instance().HasResult()) {
            std::string res = ifd::FileDialog::Instance().GetResult().u8string();
            printf("SAVE[%s]\n", res.c_str());
        }
        ifd::FileDialog::Instance().Close();
    }
*/
}
