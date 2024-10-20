#include "WindowController.h"

#include <sstream>
#include <GLFW/glfw3.h>

#include <boost/algorithm/string.hpp>

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"

#include "MainLoopEntityController.h"

namespace
{
    auto const WindowedMode = "window";
    auto const DesktopMode = "desktop";

    GLFWvidmode convert(std::string const& mode)
    {
        std::vector<std::string> modeParts;
        boost::split(modeParts, mode, [](char c) { return c == ' '; });

        CHECK(modeParts.size() == 6);

        GLFWvidmode result;
        result.width = std::stoi(modeParts.at(0));
        result.height = std::stoi(modeParts.at(1));
        result.redBits = std::stoi(modeParts.at(2));
        result.greenBits = std::stoi(modeParts.at(3));
        result.blueBits = std::stoi(modeParts.at(4));
        result.refreshRate = std::stoi(modeParts.at(5));
        return result;
    }

    std::string convert(GLFWvidmode const& vidmode)
    {
        std::vector<std::string> modeParts;
        modeParts.emplace_back(std::to_string(vidmode.width));
        modeParts.emplace_back(std::to_string(vidmode.height));
        modeParts.emplace_back(std::to_string(vidmode.redBits));
        modeParts.emplace_back(std::to_string(vidmode.greenBits));
        modeParts.emplace_back(std::to_string(vidmode.blueBits));
        modeParts.emplace_back(std::to_string(vidmode.refreshRate));

        return boost::join(modeParts, " ");
    }
}

void WindowController::init()
{
    auto& settings = GlobalSettings::get();
    _mode = settings.getValue("settings.display.mode", DesktopMode);
    _sizeInWindowedMode.x = std::max(100, settings.getValue("settings.display.window width", _sizeInWindowedMode.x));
    _sizeInWindowedMode.y = std::max(100, settings.getValue("settings.display.window height", _sizeInWindowedMode.y));
    _fps = settings.getValue("settings.display.fps", _fps);
    _lastContentScaleFactor = settings.getValue("settings.display.content scale factor", _lastContentScaleFactor);

    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    _windowData.mode = glfwGetVideoMode(primaryMonitor);

    _desktopVideoMode = std::make_shared<GLFWvidmode>();
    *_desktopVideoMode = *_windowData.mode;

    _windowData.window = [&] {
        if (isWindowedMode()) {
            log(Priority::Important, "set windowed mode");
            _startupSize = _sizeInWindowedMode;
            return glfwCreateWindow(_sizeInWindowedMode.x, _sizeInWindowedMode.y, "alien", nullptr, nullptr);
        } else {
            log(Priority::Important, "set full screen mode");
            _startupSize = {_windowData.mode->width, _windowData.mode->height};
            return glfwCreateWindow(_windowData.mode->width, _windowData.mode->height, "alien", primaryMonitor, nullptr);
        }
    }();

    if (_windowData.window == nullptr) {
        throw std::runtime_error("Failed to create window.");
    }
    glfwMakeContextCurrent(_windowData.window);

    if (!isWindowedMode() && !isDesktopMode()) {
        auto userMode = getUserDefinedResolution();
        _startupSize = {userMode.width, userMode.height};
        log(Priority::Important, "switching to  " + createLogString(userMode));
        glfwSetWindowMonitor(_windowData.window, primaryMonitor, 0, 0, userMode.width, userMode.height, userMode.refreshRate);
    }

    float temp;
    glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &_contentScaleFactor, &temp);  //consider only horizontal content scale
}

void WindowController::shutdown()
{
    if (isWindowedMode()) {
        updateWindowSize();
    }
    auto& settings = GlobalSettings::get();
    settings.setValue("settings.display.mode", _mode);
    settings.setValue("settings.display.window width", _sizeInWindowedMode.x);
    settings.setValue("settings.display.window height", _sizeInWindowedMode.y);
    settings.setValue("settings.display.fps", _fps);
    settings.setValue("settings.display.content scale factor", _contentScaleFactor);

}

auto WindowController::getWindowData() -> WindowData
{
    return _windowData;
}

bool WindowController::isWindowedMode()
{
    return _mode == WindowedMode;
}

void WindowController::setWindowedMode()
{
    setMode(WindowedMode);
}

bool WindowController::isDesktopMode()
{
    return _mode == DesktopMode;
}

void WindowController::setDesktopMode()
{
    setMode(DesktopMode);
}

GLFWvidmode WindowController::getUserDefinedResolution()
{
    return convert(_mode);
}

void WindowController::setUserDefinedResolution(GLFWvidmode const& videoMode)
{
    setMode(convert(videoMode));
}

IntVector2D WindowController::getStartupWindowSize()
{
    return _startupSize;
}

std::string WindowController::getMode()
{
    return _mode;
}

void WindowController::setMode(std::string const& mode)
{
    if (getMode() == mode) {
        return;
    }
    if (isWindowedMode()) {
        updateWindowSize();
    }

    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();

    if(mode == WindowedMode) {
        log(Priority::Important, "set windowed mode");
        GLFWvidmode const* desktopVideoMode = glfwGetVideoMode(primaryMonitor);
        glfwSetWindowMonitor(
            _windowData.window,
            nullptr,
            0,
            0,
            _sizeInWindowedMode.x,
            _sizeInWindowedMode.y,
            desktopVideoMode->refreshRate);
    } else if(mode == DesktopMode) {
        log(
            Priority::Important, "set full screen mode with " + createLogString(*_desktopVideoMode));
        glfwSetWindowMonitor(
            _windowData.window,
            primaryMonitor,
            0,
            0,
            _desktopVideoMode->width,
            _desktopVideoMode->height,
            _desktopVideoMode->refreshRate);
    } else {
        auto userMode = convert(mode);
        log(Priority::Important, "set full screen mode with " + createLogString(userMode));
        glfwSetWindowMonitor(
            _windowData.window, primaryMonitor, 0, 0, userMode.width, userMode.height, userMode.refreshRate);
    }
    _mode = mode;
}

void WindowController::updateWindowSize()
{
    glfwGetWindowSize(_windowData.window, &_sizeInWindowedMode.x, &_sizeInWindowedMode.y);
}

std::string WindowController::createLogString(GLFWvidmode const& videoMode)
{
    std::stringstream ss;
    ss << videoMode.width << " x " << videoMode.height << " @ " << videoMode.refreshRate << "Hz";
    return ss.str();
}

int WindowController::getFps()
{
    return _fps;
}

void WindowController::setFps(int value)
{
    _fps = value;
}

float WindowController::getContentScaleFactor()
{
    return _contentScaleFactor;
}

float WindowController::getLastContentScaleFactor()
{
    return _lastContentScaleFactor;
}
