#pragma once

#include <optional>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class WindowController : public MainLoopEntity
{
    MAKE_SINGLETON(WindowController);

public:
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

    void updateWindowTitle(std::string const& projectName);

    int getFps();
    void setFps(int value);

    float getContentScaleFactor();
    float getContentScaleCorrection();

    float getOsContentScaleFactor();

    // Configured content scale factor. It becomes the content scale factor of the session at the next startup.
    float getConfiguredContentScaleFactor();
    bool isAutoContentScaleFactor();
    void setAutoContentScaleFactor(bool value);
    float getUserDefinedContentScaleFactor();
    void setUserDefinedContentScaleFactor(float value);

    static auto constexpr MinContentScaleFactor = 1.0f;
    static auto constexpr MaxContentScaleFactor = 4.0f;

private:
    void init() override;
    void process() override {}
    void shutdown() override;

    void updateWindowSize();
    std::string createLogString(GLFWvidmode const& videoMode);

    WindowData _windowData;
    std::shared_ptr<GLFWvidmode> _desktopVideoMode;
    IntVector2D _startupSize;
    IntVector2D _sizeInWindowedMode = {1920 * 3 / 4, 1080 * 3 / 4};
    float _contentScaleFactor = 1.0f;
    std::optional<float> _lastContentScaleFactor;
    float _osContentScaleFactor = 1.0f;
    float _userDefinedContentScaleFactor = 0.0f;
    bool _autoContentScaleFactor = true;
    int _fps = 33;

    std::string _mode;
    std::string _lastProjectName;
};
