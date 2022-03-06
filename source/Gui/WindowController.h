#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"

class _WindowController
{
public:
    _WindowController();
    ~_WindowController();

    void shutdown();

    struct WindowData
    {
        GLFWwindow* window;
        GLFWvidmode const* mode;
    };
    WindowData getWindowData() const;

    bool isWindowedMode() const;
    void setWindowedMode();

    bool isDesktopMode() const;
    void setDesktopMode();

    GLFWvidmode getUserDefinedResolution() const;
    void setUserDefinedResolution(GLFWvidmode const& videoMode);

    IntVector2D getStartupWindowSize() const;

    std::string getMode() const;
    void setMode(std::string const& mode);

    int getFps() const;
    void setFps(int value);

private:

    void updateWindowSize();
    std::string createLogString(GLFWvidmode const& videoMode) const;

    WindowData _windowData;
    GLFWvidmode* _desktopVideoMode;
    IntVector2D _startupSize;
    IntVector2D _sizeInWindowedMode = {1920 * 3 / 4, 1080 * 3 / 4};
    int _fps = 40;

    std::string _mode;
};