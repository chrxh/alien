#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"

class WindowController
{
public:
    WindowController() = delete;

    static void init();
    static void shutdown();

    struct WindowData
    {
        GLFWwindow* window;
        GLFWvidmode const* mode;
    };
    static WindowData getWindowData();

    static bool isWindowedMode();
    static void setWindowedMode();

    static bool isDesktopMode();
    static void setDesktopMode();

    static GLFWvidmode getUserDefinedResolution();
    static void setUserDefinedResolution(GLFWvidmode const& videoMode);

    static IntVector2D getStartupWindowSize();

    static std::string getMode();
    static void setMode(std::string const& mode);

    static int getFps();
    static void setFps(int value);

    static float getContentScaleFactor();
    static float getLastContentScaleFactor();

private:
    static void updateWindowSize();
    static std::string createLogString(GLFWvidmode const& videoMode);

    static WindowData _windowData;
    static std::shared_ptr<GLFWvidmode> _desktopVideoMode;
    static IntVector2D _startupSize;
    static IntVector2D _sizeInWindowedMode;
    static float _contentScaleFactor;
    static float _lastContentScaleFactor;
    static int _fps;

    static std::string _mode;
};