#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"

class _WindowController
{
public:
    _WindowController();
    ~_WindowController();

    struct WindowData
    {
        GLFWwindow* window;
        GLFWvidmode const* mode;
    };
    WindowData getWindowData() const;

    bool isFullscreen() const;
    void setFullscreen(bool value);

    IntVector2D getStartupWindowSize() const;

private:
    std::string createVideoModeString(GLFWvidmode const& videoMode) const;

    IntVector2D _defaultSize;
    WindowData _displayData;
    bool _fullscreen = true;
    bool _useDesktopResolution = true; 

    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes = nullptr;
    int _videoModeSelection = 0;
};