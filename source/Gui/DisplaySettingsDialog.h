#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _DisplaySettingsDialog
{
public:
    _DisplaySettingsDialog(GLFWwindow* window);
    ~_DisplaySettingsDialog();

    void process();

    void show();

private:
    void onSetVideoMode();

    int getOptimalVideoModeIndex() const;
    std::string createVideoModeString(GLFWvidmode const& videoMode) const;
    std::vector<std::string> createVideoModeStrings() const;

    GLFWwindow* _window = nullptr;
    GLFWvidmode const* _desktopVideoMode = nullptr;
    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes = nullptr;   //will be extended at front by 1 entry(= desktop)

    bool _show = false;

    //0 = desktop and 1, ..., _videoModesCount = possible video modes
    int _videoModeSelection = 0;
    int _origVideoModeSelection = 0;  
};