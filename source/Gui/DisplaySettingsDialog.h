#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _DisplaySettingsDialog
{
public:
    _DisplaySettingsDialog(WindowController const& windowController);
    ~_DisplaySettingsDialog();

    void process();
    void show();

private:
    void setFullscreen(int selectionIndex);
    int getSelectionIndex() const;
    std::vector<std::string> createVideoModeStrings() const;

    WindowController _windowController;

    bool _show = false;
    std::string _origMode;
    int _origSelectionIndex;
    int _selectionIndex;
    int _origFps;

    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes;
    std::vector<std::string> _videoModeStrings;
};