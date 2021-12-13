#include "WindowController.h"

#include <GLFW/glfw3.h>

#include "Base/LoggingService.h"
#include "Base/ServiceLocator.h"

#include "GlobalSettings.h"

_WindowController::_WindowController()
{
    _fullscreen = GlobalSettings::getInstance().getBoolState("settings.display.full screen", _fullscreen);
    _useDesktopResolution =
        GlobalSettings::getInstance().getBoolState("settings.display.use desktop resolution", _useDesktopResolution);
    _videoModeSelection = GlobalSettings::getInstance().getIntState("settings.display.video mode", _videoModeSelection);
    _sizeInWindowedMode.x = GlobalSettings::getInstance().getIntState("settings.display.window width", _sizeInWindowedMode.x);
    _sizeInWindowedMode.y = GlobalSettings::getInstance().getIntState("settings.display.window height", _sizeInWindowedMode.y);

    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    _videoModes = glfwGetVideoModes(primaryMonitor, &_videoModesCount);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    _windowData.mode = glfwGetVideoMode(primaryMonitor);
    auto screenWidth = _windowData.mode->width;
    auto screenHeight = _windowData.mode->height;

    _windowData.window = [&] {
        if (_fullscreen) {
            loggingService->logMessage(
                Priority::Important, "set full screen mode");
            return glfwCreateWindow(screenWidth, screenHeight, "alien", primaryMonitor, NULL);
        } else {
            loggingService->logMessage(Priority::Important, "set window mode");
            return glfwCreateWindow(_sizeInWindowedMode.x, _sizeInWindowedMode.y, "alien", NULL, NULL);
        }
    }();
    if (_windowData.window == NULL) {
        throw std::runtime_error("Failed to create window.");
    }
    glfwMakeContextCurrent(_windowData.window);

    if (_fullscreen) {
        if (_useDesktopResolution) {
            auto desktopVideoMode = glfwGetVideoMode(primaryMonitor);
            _statupSize = {desktopVideoMode->width, desktopVideoMode->height};
            loggingService->logMessage(
                Priority::Important, "use desktop resolution with " + createVideoModeString(*desktopVideoMode));
        } else {
            auto mode = _videoModes[_videoModeSelection];
            _statupSize = {mode.width, mode.height};
            loggingService->logMessage(Priority::Important, "switching to  " + createVideoModeString(mode));
            glfwSetWindowMonitor(_windowData.window, primaryMonitor, 0, 0, mode.width, mode.height, mode.refreshRate);
        }
    } else {
        _statupSize = _sizeInWindowedMode;
    }
}

_WindowController::~_WindowController()
{
}

void _WindowController::shutdown()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setBoolState("settings.display.full screen", _fullscreen);
    settings.setBoolState("settings.display.use desktop resolution", _useDesktopResolution);
    settings.setIntState("settings.display.video mode", _videoModeSelection);
    if (!_fullscreen) {
        updateWindowSize();
    }
    settings.setIntState("settings.display.window width", _sizeInWindowedMode.x);
    settings.setIntState("settings.display.window height", _sizeInWindowedMode.y);
}

auto _WindowController::getWindowData() const -> WindowData
{
    return _windowData;
}

bool _WindowController::isFullscreen() const
{
    return _fullscreen;
}

void _WindowController::setFullscreen(bool value)
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    if (value) {
        updateWindowSize(); //switching from windowed to full screen mode => save window size

        loggingService->logMessage(Priority::Important, "set full screen mode");
        if (_useDesktopResolution) {
            GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
            GLFWvidmode const* desktopVideoMode = glfwGetVideoMode(primaryMonitor);
            loggingService->logMessage(
                Priority::Important, "use desktop resolution with " + createVideoModeString(*desktopVideoMode));
            glfwSetWindowMonitor(
                _windowData.window,
                primaryMonitor,
                0,
                0,
                desktopVideoMode->width,
                desktopVideoMode->height,
                desktopVideoMode->refreshRate);
        }
    } else {
        loggingService->logMessage(Priority::Important, "set windowed mode");
        GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
        GLFWvidmode const* desktopVideoMode = glfwGetVideoMode(primaryMonitor);
        glfwSetWindowMonitor(
            _windowData.window,
            NULL,
            0,
            0,
            _sizeInWindowedMode.x,
            _sizeInWindowedMode.y,
            desktopVideoMode->refreshRate);
    }
    _fullscreen = value;
}

IntVector2D _WindowController::getStartupWindowSize() const
{
    return _statupSize;
}

void _WindowController::updateWindowSize()
{
    glfwGetWindowSize(_windowData.window, &_sizeInWindowedMode.x, &_sizeInWindowedMode.y);
}

std::string _WindowController::createVideoModeString(GLFWvidmode const& videoMode) const
{
    std::stringstream ss;
    ss << videoMode.width << " x " << videoMode.height << " @ " << videoMode.refreshRate << "Hz";
    return ss.str();
}
