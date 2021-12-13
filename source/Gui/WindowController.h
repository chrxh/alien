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

    bool isFullscreen() const;
    void setFullscreen(bool value);

    IntVector2D getStartupWindowSize() const;

private:
    void updateWindowSize();
    std::string createVideoModeString(GLFWvidmode const& videoMode) const;

    WindowData _windowData;
    IntVector2D _statupSize;
    IntVector2D _sizeInWindowedMode = {1920 * 3 / 4, 1080 * 3 / 4};

    bool _fullscreen = true;
    bool _useDesktopResolution = true; 

    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes = nullptr;
    int _videoModeSelection = 0;
};