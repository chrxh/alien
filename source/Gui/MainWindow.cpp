#include "MainWindow.h"

#include <iostream>

#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include <Fonts/IconsFontAwesome5.h>

#include <Base/AlienExceptions.h>
#include <Base/GlobalSettings.h>
#include <Base/Resources.h>

#include <Network/NetworkService.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/SerializerService.h>

#include "AboutDialog.h"
#include "ActivateUserDialog.h"
#include "AlienGui.h"
#include "AutosaveController.h"
#include "AutosaveWindow.h"
#include "BrowserController.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "DelayedExecutionController.h"
#include "DeleteUserDialog.h"
#include "DisplaySettingsDialog.h"
#include "DocumentationWindow.h"
#include "EditSimulationDialog.h"
#include "EditorController.h"
#include "EvolutionDashboardWindow.h"
#include "ExitDialog.h"
#include "FileTransferController.h"
#include "FpsController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
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
#include "McpController.h"
#include "McpSettingsDialog.h"
#include "McpWindow.h"
#include "NetworkSettingsDialog.h"
#include "NetworkTransferController.h"
#include "NewPasswordDialog.h"
#include "NewSimulationDialog.h"
#include "OverlayController.h"
#include "PreviewSettingsDialog.h"
#include "ReplaceSimulationDialog.h"
#include "ResetPasswordDialog.h"
#include "SavePictureDialog.h"
#include "SignalsBufferDialog.h"
#include "SimulationInteractionController.h"
#include "SimulationParametersMainWindow.h"
#include "SimulationView.h"
#include "SpatialControlWindow.h"
#include "StartupCheckService.h"
#include "StyleService.h"
#include "TemporalControlWindow.h"
#include "TextureService.h"
#include "UiController.h"
#include "UploadSimulationDialog.h"
#include "Viewport.h"
#include "VulkanContext.h"
#include "VulkanFrameRenderer.h"
#include "VulkanGeometryBuffers.h"
#include "WindowController.h"
#include "implot.h"

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
        }
    }
}

_MainWindow::_MainWindow()
{
    IMGUI_CHECKVERSION();

    StartupCheckService::get().check();

    log(Priority::Important, "initialize GLFW and Vulkan");
    initGlfwAndVulkan();

    LogWindow::get().setup();

    log(Priority::Important, "initialize services");
    StyleService::get().setup();
    NetworkService::get().setup();

    log(Priority::Important, "initialize facades");
    _PersisterFacade::get()->setup();

    log(Priority::Important, "initialize main loop elements");
    Viewport::get().setup();
    EditorController::get().setup();
    SimulationView::get().setup();
    SimulationInteractionController::get().setup();
    EvolutionDashboardWindow::get().setup();
    TemporalControlWindow::get().setup();
    SpatialControlWindow::get().setup();
    SimulationParametersMainWindow::get().setup();
    LocationController::get().setup();
    MainLoopController::get().setup();
    ExitDialog::get().setup();
    MassOperationsDialog::get().setup();
    DocumentationWindow::get().setup();
    NewSimulationDialog::get().setup();
    BrowserController::get().setup();
    BrowserWindow::get().setup();
    ActivateUserDialog::get().setup();
    NewPasswordDialog::get().setup();
    LoginDialog::get().setup();
    UploadSimulationDialog::get().setup();
    ReplaceSimulationDialog::get().setup();
    ImageToPatternDialog::get().setup();
    AutosaveController::get().setup();
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
    PreviewSettingsDialog::get().setup();
    ResetPasswordDialog::get().setup();
    GenericMessageDialog::get().setup();
    GenericFileDialog::get().setup();
    SavePictureDialog::get().setup();
    SignalsBufferDialog::get().setup();
    DelayedExecutionController::get().setup();
    UiController::get().setup();
    McpController::get().setup();
    McpSettingsDialog::get().setup();
    McpWindow::get().setup();

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

    auto window = WindowController::get().getWindowData().window;
    glfwHideWindow(window);

    // The simulation releases its imports of the geometry buffers before Vulkan frees them
    _PersisterFacade::get()->shutdown();
    _SimulationFacade::get()->closeSimulation();

    NetworkService::get().shutdown();

    auto& vulkanContext = VulkanContext::get();
    vulkanContext.waitIdle();
    SimulationView::get().releaseGraphicsResources();
    vulkanContext.releasePendingResources();
    TextureService::get().shutdown();
    VulkanFrameRenderer::get().shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    vulkanContext.shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();

    log(Priority::Important, "user interface shut down");
}

namespace
{
    // Errors of the GPU engine are reported when the simulation is created
    std::optional<GpuUuid> getEngineGpuUuid()
    {
        try {
            return _SimulationFacade::get()->getGpuUuid();
        } catch (std::exception const&) {
            return std::nullopt;
        }
    }

    // Geometry buffers created afterwards are only shareable with the GPU engine if the check succeeds
    void checkRenderingInterop()
    {
        if (!GlobalSettings::get().isInterop() || !VulkanContext::get().isMemorySharingSupported()) {
            return;
        }
        // Without objects, the buffers get their minimum capacity
        auto geometryBuffers = _VulkanGeometryBuffers::create();
        geometryBuffers->updateNumObjects({});
        if (_SimulationFacade::get()->isRenderingInteropWorking(geometryBuffers)) {
            log(Priority::Important, "CUDA-Vulkan interop is working");
        } else {
            GlobalSettings::get().setInterop(false);
            log(Priority::Important, "CUDA-Vulkan interop is not working on this system, falling back to the transfer over host memory");
        }
    }
}

void _MainWindow::initGlfwAndVulkan()
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize Glfw.");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    WindowController::get().setup();
    auto windowData = WindowController::get().getWindowData();
    glfwSetFramebufferSizeCallback(windowData.window, framebufferSizeCallback);

    VulkanContext::get().setup(windowData.window, getEngineGpuUuid());
    checkRenderingInterop();

    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(windowData.window, true);
    VulkanFrameRenderer::get().setup(windowData.window);
}

void _MainWindow::initFileDialogs()
{
    ifd::FileDialog::Instance().CreateTexture = [](uint8_t* data, int w, int h, char fmt) -> void* {
        auto texture = TextureService::get().createTexture(data, w, h, fmt == 0 ? TextureFormat::Bgra : TextureFormat::Rgba, TextureFilter::Nearest);
        return reinterpret_cast<void*>(static_cast<uintptr_t>(texture.textureId));
    };
    ifd::FileDialog::Instance().DeleteTexture = [](void* texture) {
        TextureService::get().deleteTexture(static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(texture)));
    };
}
