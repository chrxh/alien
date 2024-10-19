#pragma once

#include "Base/Definitions.h"
#include "Base/Singleton.h"

#include "Definitions.h"

class WindowController
{
    MAKE_SINGLETON(WindowController);

public:
    void init();
    void shutdown();

    struct WindowData
    {
        GLFWwindow* window;
        GLFWvidmode const* mode;
    };
    WindowData getWindowData();

    bool isWindowedMode();
    void setWindowedMode();

    bool isDesktopMode();
    void setDesktopMode();

    GLFWvidmode getUserDefinedResolution();
    void setUserDefinedResolution(GLFWvidmode const& videoMode);

    IntVector2D getStartupWindowSize();

    std::string getMode();
    void setMode(std::string const& mode);

    int getFps();
    void setFps(int value);

    float getContentScaleFactor();
    float getLastContentScaleFactor();

private:
    void updateWindowSize();
    std::string createLogString(GLFWvidmode const& videoMode);

    WindowData _windowData;
    std::shared_ptr<GLFWvidmode> _desktopVideoMode;
    IntVector2D _startupSize;
    IntVector2D _sizeInWindowedMode = {1920 * 3 / 4, 1080 * 3 / 4};
    float _contentScaleFactor = 1.0f;
    float _lastContentScaleFactor = 1.0f;
    int _fps = 33;

    std::string _mode;
};