#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"

class WindowController
{
public:
    static WindowController& getInstance();
    ~WindowController();

    void init();
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

    float getContentScaleFactor() const;
    float getLastContentScaleFactor() const;

private:
    WindowController();

    void updateWindowSize();
    std::string createLogString(GLFWvidmode const& videoMode) const;

    WindowData _windowData;
    std::shared_ptr<GLFWvidmode> _desktopVideoMode;
    IntVector2D _startupSize;
    IntVector2D _sizeInWindowedMode = {1920 * 3 / 4, 1080 * 3 / 4};
    float _contentScaleFactor = 1.0f;
    float _lastContentScaleFactor = 1.0f;
    int _fps = 40;

    std::string _mode;
};