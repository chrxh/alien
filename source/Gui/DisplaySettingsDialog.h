#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _DisplaySettingsDialog
{
public:
    _DisplaySettingsDialog();
    ~_DisplaySettingsDialog();

    void process();
    void show();

private:
    void setFullscreen(int selectionIndex);
    int getSelectionIndex() const;
    std::vector<std::string> createVideoModeStrings() const;

    bool _show = false;
    std::string _origMode;
    int _origSelectionIndex;
    int _selectionIndex;
    int _origFps;

    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes;
    std::vector<std::string> _videoModeStrings;
};