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
    bool isDesktopMode() const;
    GLFWvidmode getUserDefinedResolution() const;

    void setWindowedMode();
    void setDesktopMode();
    void setUserDefinedResolution(GLFWvidmode const& videoMode);

    IntVector2D getStartupWindowSize() const;

    std::string getMode() const;
    void setMode(std::string const& mode);

private:

    void updateWindowSize();
    std::string createLogString(GLFWvidmode const& videoMode) const;

    WindowData _windowData;
    GLFWvidmode* _desktopVideoMode;
    IntVector2D _startupSize;
    IntVector2D _sizeInWindowedMode = {1920 * 3 / 4, 1080 * 3 / 4};

    std::string _mode;

/*
    bool _fullscreen = true;
    bool _useDesktopResolution = true; 
*/

//    std::vector<GLFWvidmode const> _videoModes;
//    int _selectionIndex = 0;   //0 = windowed mode, 1 = full screen with desktop resolution, 2 ... n+2 = video mode n
};