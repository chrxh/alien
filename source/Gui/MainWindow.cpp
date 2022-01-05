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
#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "EngineInterface/Serializer.h"
#include "EngineImpl/SimulationController.h"

#include "ModeWindow.h"
#include "SimulationView.h"
#include "StyleRepository.h"
#include "TemporalControlWindow.h"
#include "SpatialControlWindow.h"
#include "SimulationParametersWindow.h"
#include "StatisticsWindow.h"
#include "GpuSettingsDialog.h"
#include "Viewport.h"
#include "NewSimulationDialog.h"
#include "StartupWindow.h"
#include "FlowGeneratorWindow.h"
#include "AlienImGui.h"
#include "AboutDialog.h"
#include "ColorizeDialog.h"
#include "LogWindow.h"
#include "SimpleLogger.h"
#include "UiController.h"
#include "GlobalSettings.h"
#include "AutosaveController.h"
#include "GettingStartedWindow.h"
#include "OpenSimulationDialog.h"
#include "SaveSimulationDialog.h"
#include "DisplaySettingsDialog.h"
#include "EditorController.h"
#include "SelectionWindow.h"
#include "ManipulatorWindow.h"
#include "WindowController.h"

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

_MainWindow::_MainWindow(SimulationController const& simController, SimpleLogger const& logger)
{
    _logger = logger;
    _simController = simController;
    
    auto glfwVersion = initGlfw();

    _windowController = boost::make_shared<_WindowController>();

    auto windowData = _windowController->getWindowData();
    glfwSetFramebufferSizeCallback(windowData.window, framebuffer_size_callback);
    glfwSwapInterval(1);  //enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();

//     ImGui::StyleColorsDark();
//     ImGui::StyleColorsLight();

    StyleRepository::getInstance().init();

    // Setup Platform/Renderer back-ends
    ImGui_ImplGlfw_InitForOpenGL(windowData.window, true);
    ImGui_ImplOpenGL3_Init(glfwVersion);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    auto worldSize = _simController->getWorldSize();
    _viewport = boost::make_shared<_Viewport>(_windowController);
    _uiController = boost::make_shared<_UiController>();
    _autosaveController = boost::make_shared<_AutosaveController>(_simController);

    _editorController =
        boost::make_shared<_EditorController>(_simController, _viewport);
    _modeWindow = boost::make_shared<_ModeWindow>(_editorController);
    _simulationView = boost::make_shared<_SimulationView>(_simController, _modeWindow, _viewport);
    simulationViewPtr = _simulationView.get();
    _statisticsWindow = boost::make_shared<_StatisticsWindow>(_simController);
    _temporalControlWindow = boost::make_shared<_TemporalControlWindow>(_simController, _statisticsWindow);
    _spatialControlWindow = boost::make_shared<_SpatialControlWindow>(_simController, _viewport);
    _simulationParametersWindow = boost::make_shared<_SimulationParametersWindow>(_simController);
    _gpuSettingsDialog = boost::make_shared<_GpuSettingsDialog>(_simController);
    _newSimulationDialog = boost::make_shared<_NewSimulationDialog>(_simController, _viewport, _statisticsWindow);
    _startupWindow = boost::make_shared<_StartupWindow>(_simController, _viewport);
    _flowGeneratorWindow = boost::make_shared<_FlowGeneratorWindow>(_simController);
    _aboutDialog = boost::make_shared<_AboutDialog>();
    _colorizeDialog = boost::make_shared<_ColorizeDialog>(_simController);
    _logWindow = boost::make_shared<_LogWindow>(_logger);
    _gettingStartedWindow = boost::make_shared<_GettingStartedWindow>();
    _openSimulationDialog = boost::make_shared<_OpenSimulationDialog>(_simController, _statisticsWindow, _viewport);
    _saveSimulationDialog = boost::make_shared<_SaveSimulationDialog>(_simController);
    _displaySettingsDialog = boost::make_shared<_DisplaySettingsDialog>(_windowController);

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
}

void _MainWindow::mainLoop()
{
    bool show_demo_window = true;
    while (!glfwWindowShouldClose(_window) && !_onClose)
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

/*
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);
        {}
*/

        switch (_startupWindow->getState()) {
        case _StartupWindow::State::Unintialized:
            processUninitialized();
            break;
        case _StartupWindow::State::RequestLoading:
            processRequestLoading();
            break;
        case _StartupWindow::State::LoadingSimulation:
            processLoadingSimulation();
            break;
        case _StartupWindow::State::LoadingControls:
            processLoadingControls();
            break;
        case _StartupWindow::State::FinishedLoading:
            processFinishedLoading();
            break;
        default:
            THROW_NOT_IMPLEMENTED();
        }
    }
}

void _MainWindow::shutdown()
{
    _windowController->shutdown();
    _autosaveController->shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(_window);
    glfwTerminate();

    _simulationView.reset();
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
    _startupWindow->process();

    // render content
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
    _startupWindow->process();
    renderSimulation();
}

void _MainWindow::processLoadingSimulation()
{
    _startupWindow->process();
    renderSimulation();
}

void _MainWindow::processLoadingControls()
{
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, Const::SliderBarWidth);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10);

    processMenubar();
    processDialogs();
    processWindows();
    processControllers();

    _uiController->process();
    _simulationView->processControls();
    _startupWindow->process();

    ImGui::PopStyleVar(2);

    renderSimulation();
}

void _MainWindow::processFinishedLoading()
{
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, Const::SliderBarWidth);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10);

    processMenubar();
    processDialogs();

/*
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.55f, 0.4f, 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.55f, 0.4f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.55f, 0.4f, 0.7f));
*/
    processWindows();
/*
    ImGui::PopStyleColor(3);
*/
    processControllers();
    _uiController->process();
    _simulationView->processControls();

    ImGui::PopStyleVar(2);

    renderSimulation();
}

void _MainWindow::renderSimulation()
{
    int display_w, display_h;
    glfwGetFramebufferSize(_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    if (_renderSimulation) {
        _simulationView->processContent();
    } else {
        glClearColor(0, 0, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    ImGui::Render();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(_window);
}

void _MainWindow::processMenubar()
{
    auto selectionWindow = _editorController->getSelectionWindow();
    auto manipulatorWindow = _editorController->getManipulatorWindow();

    if (ImGui::BeginMainMenuBar()) {
        if (AlienImGui::ShutdownButton()) {
            _showExitDialog = true;
        }
        ImGui::Dummy(ImVec2(10.0f, 0.0f));
        if (AlienImGui::BeginMenuButton(" " ICON_FA_GAMEPAD "  Simulation ", _simulationMenuToggled, "Simulation")) {
            if (ImGui::MenuItem("New", "CTRL+N")) {
                _newSimulationDialog->show();
                _simulationMenuToggled = false;
            }
            if (ImGui::MenuItem("Open", "CTRL+O")) {
                _openSimulationDialog->show();
                _simulationMenuToggled = false;
            }
            if (ImGui::MenuItem("Save", "CTRL+S")) {
                _saveSimulationDialog->show();
                _simulationMenuToggled = false;
            }
            ImGui::Separator();
            ImGui::BeginDisabled(_simController->isSimulationRunning());
            if (ImGui::MenuItem("Run", "CTRL+R")) {
                onRunSimulation();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!_simController->isSimulationRunning());
            if (ImGui::MenuItem("Pause", "CTRL+P")) {
                onPauseSimulation();
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
            if (ImGui::MenuItem("Flow generator", "ALT+5", _flowGeneratorWindow->isOn())) {
                _flowGeneratorWindow->setOn(!_flowGeneratorWindow->isOn());
            }
            if (ImGui::MenuItem("Log", "ALT+6", _logWindow->isOn())) {
                _logWindow->setOn(!_logWindow->isOn());
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_PEN_ALT "  Editor ", _editorMenuToggled, "Editor")) {
            if (ImGui::MenuItem("Activate", "ALT+E", _modeWindow->getMode() == _ModeWindow::Mode::Action)) {
                _modeWindow->setMode(
                    _modeWindow->getMode() == _ModeWindow::Mode::Action ? _ModeWindow::Mode::Navigation
                                                                        : _ModeWindow::Mode::Action);
            }
            ImGui::BeginDisabled(_ModeWindow::Mode::Navigation == _modeWindow->getMode());
            if (ImGui::MenuItem("Selection", "ALT+S", selectionWindow->isOn())) {
                selectionWindow->setOn(!selectionWindow->isOn());
            }
            if (ImGui::MenuItem("Creator", "ALT+C", true)) {
            }
            if (ImGui::MenuItem("Manipulator", "ALT+M", manipulatorWindow->isOn())) {
                manipulatorWindow->setOn(!manipulatorWindow->isOn());
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(
                _ModeWindow::Mode::Navigation == _modeWindow->getMode()
                || !_editorController->isInspectionPossible());
            if (ImGui::MenuItem("Inspect entities", "ALT+N")) {
                _editorController->onInspectEntities();
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(
                _ModeWindow::Mode::Navigation == _modeWindow->getMode()
                || !_editorController->areInspectionWindowsActive());
            if (ImGui::MenuItem("Close inspections", "ESC")) {
                _editorController->onCloseAllInspectorWindows();
            }
            ImGui::EndDisabled();
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_COG "  Settings ", _settingsMenuToggled, "Settings")) {
            if (ImGui::MenuItem("Auto save", "", _autosaveController->isOn())) {
                _autosaveController->setOn(!_autosaveController->isOn());
            }
            if (ImGui::MenuItem("GPU settings", "ALT+C")) {
                _gpuSettingsDialog->show();
            }
            if (ImGui::MenuItem("Display settings", "ALT+V")) {
                _displaySettingsDialog->show();
            }
            AlienImGui::EndMenuButton();
        }
        if (AlienImGui::BeginMenuButton(" " ICON_FA_EYE "  View ", _viewMenuToggled, "View")) {
            if (ImGui::MenuItem("Information overlay", "ALT+O", _simulationView->isOverlayActive())) {
                _simulationView->setOverlayActive(!_simulationView->isOverlayActive());
            }
            if (ImGui::MenuItem("Render UI", "ALT+U", _uiController->isOn())) {
                _uiController->setOn(!_uiController->isOn());
            }
            if (ImGui::MenuItem("Render Simulation", "ALT+I", _renderSimulation)) {
                _renderSimulation = !_renderSimulation;
            }
            AlienImGui::EndMenuButton();
        }

        if (AlienImGui::BeginMenuButton(" " ICON_FA_TOOLS "  Tools ", _toolsMenuToggled, "Tools")) {
            if (ImGui::MenuItem("Colorize", "ALT+H")) {
                _colorizeDialog->show();
                _toolsMenuToggled = false;
            }
            AlienImGui::EndMenuButton();
        }
        if (AlienImGui::BeginMenuButton(" " ICON_FA_LIFE_RING "  Help ", _helpMenuToggled, "Help")) {
            if (ImGui::MenuItem("About", "")) {
                _aboutDialog->show();
                _helpMenuToggled = false;
            }
            if (ImGui::MenuItem("Getting started", "", _gettingStartedWindow->isOn())) {
                _gettingStartedWindow->setOn(!_gettingStartedWindow->isOn());
            }
            AlienImGui::EndMenuButton();
        }
        ImGui::EndMainMenuBar();
    }

    //menu hotkeys
    auto io = ImGui::GetIO();
    if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_N)) {
        _newSimulationDialog->show();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_O)) {
        _openSimulationDialog->show();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_S)) {
        _saveSimulationDialog->show();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_R)) {
        onRunSimulation();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(GLFW_KEY_P)) {
        onPauseSimulation();
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
        _flowGeneratorWindow->setOn(!_flowGeneratorWindow->isOn());
    }
    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_6)) {
        _logWindow->setOn(!_logWindow->isOn());
    }

    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_E)) {
        _modeWindow->setMode(
            _modeWindow->getMode() == _ModeWindow::Mode::Action ? _ModeWindow::Mode::Navigation
                                                                : _ModeWindow::Mode::Action);
    }
    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_S)) {
        selectionWindow->setOn(!selectionWindow->isOn());
    }
    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_M)) {
        manipulatorWindow->setOn(!manipulatorWindow->isOn());
    }
    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_N) && _editorController->isInspectionPossible()) {
        _editorController->onInspectEntities();
    }
    if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE)) {
        _editorController->onCloseAllInspectorWindows();
    }

    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_C)) {
        _gpuSettingsDialog->show();
    }
    if (io.KeyAlt && ImGui::IsKeyPressed(GLFW_KEY_V)) {
        _displaySettingsDialog->show();
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
        _colorizeDialog->show();
    }
}

void _MainWindow::processDialogs()
{
    _openSimulationDialog->process();
    _saveSimulationDialog->process();
    _newSimulationDialog->process();
    _aboutDialog->process();
    _colorizeDialog->process();
    _gpuSettingsDialog->process();
    _displaySettingsDialog->process(); 
    processExitDialog();
}

void _MainWindow::processWindows()
{
    _temporalControlWindow->process();
    _spatialControlWindow->process();
    _modeWindow->process();
    _statisticsWindow->process();
    _simulationParametersWindow->process();
    _flowGeneratorWindow->process();
    _logWindow->process();
    _gettingStartedWindow->process();
}

void _MainWindow::processControllers()
{
    _autosaveController->process();
    _editorController->process();
}

void _MainWindow::onRunSimulation()
{
    _simController->runSimulation();
}

void _MainWindow::onPauseSimulation()
{
    _simController->pauseSimulation();
}

void _MainWindow::processExitDialog()
{
     if (_showExitDialog) {
        auto name = "Exit";
        ImGui::OpenPopup(name);

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal(name, NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Do you really want to terminate the program?");

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::Spacing();

            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
                _onClose = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
                _showExitDialog = false;
            }
            ImGui::SetItemDefaultFocus();

            ImGui::EndPopup();
        }
    }
}

void _MainWindow::reset()
{
    _statisticsWindow->reset();
}
