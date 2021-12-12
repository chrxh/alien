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
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    _videoModes = glfwGetVideoModes(primaryMonitor, &_videoModesCount);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    _displayData.mode = glfwGetVideoMode(primaryMonitor);
    auto screenWidth = _displayData.mode->width;
    auto screenHeight = _displayData.mode->height;

    _displayData.window = [&] {
        if (_fullscreen) {
            loggingService->logMessage(
                Priority::Important, "set full screen mode");
            return glfwCreateWindow(screenWidth, screenHeight, "alien", primaryMonitor, NULL);
        } else {
            loggingService->logMessage(Priority::Important, "set window mode");
            _defaultSize = {screenWidth * 3 / 4, screenHeight * 3 / 4};
            return glfwCreateWindow(_defaultSize.x, _defaultSize.y, "alien", NULL, NULL);
        }
    }();
    if (_displayData.window == NULL) {
        throw std::runtime_error("Failed to create window.");
    }
    glfwMakeContextCurrent(_displayData.window);

    if (_fullscreen) {
        if (_useDesktopResolution) {
            auto desktopVideoMode = glfwGetVideoMode(primaryMonitor);
            _defaultSize = {desktopVideoMode->width, desktopVideoMode->height};
            loggingService->logMessage(
                Priority::Important, "use desktop resolution with " + createVideoModeString(*desktopVideoMode));
        } else {
            auto mode = _videoModes[_videoModeSelection];
            _defaultSize = {mode.width, mode.height};
            loggingService->logMessage(Priority::Important, "switching to  " + createVideoModeString(mode));
            glfwSetWindowMonitor(_displayData.window, primaryMonitor, 0, 0, mode.width, mode.height, mode.refreshRate);
        }
    }
}

_WindowController::~_WindowController()
{
    GlobalSettings::getInstance().setBoolState("settings.display.full screen", _fullscreen);
    GlobalSettings::getInstance().setBoolState("settings.display.use desktop resolution", _useDesktopResolution);
    GlobalSettings::getInstance().setIntState("settings.display.video mode", _videoModeSelection);
}

auto _WindowController::getWindowData() const -> WindowData
{
    return _displayData;
}

bool _WindowController::isFullscreen() const
{
    return _fullscreen;
}

void _WindowController::setFullscreen(bool value)
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    if (value) {
        loggingService->logMessage(Priority::Important, "set full screen mode");
        if (_useDesktopResolution) {
            GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
            GLFWvidmode const* desktopVideoMode = glfwGetVideoMode(primaryMonitor);
            loggingService->logMessage(
                Priority::Important, "use desktop resolution with " + createVideoModeString(*desktopVideoMode));
            glfwSetWindowMonitor(
                _displayData.window,
                primaryMonitor,
                0,
                0,
                desktopVideoMode->width,
                desktopVideoMode->height,
                desktopVideoMode->refreshRate);
        }
    } else {
        loggingService->logMessage(Priority::Important, "set window mode");
        GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
        GLFWvidmode const* desktopVideoMode = glfwGetVideoMode(primaryMonitor);
        glfwSetWindowMonitor(
            _displayData.window,
            NULL,
            desktopVideoMode->width / 8,
            desktopVideoMode->height / 8,
            desktopVideoMode->width * 3 / 4,
            desktopVideoMode->height * 3 / 4,
            desktopVideoMode->refreshRate);
    }
    _fullscreen = value;
}

IntVector2D _WindowController::getStartupWindowSize() const
{
    return _defaultSize;
}

std::string _WindowController::createVideoModeString(GLFWvidmode const& videoMode) const
{
    std::stringstream ss;
    ss << videoMode.width << " x " << videoMode.height << " @ " << videoMode.refreshRate << "Hz";
    return ss.str();
}
