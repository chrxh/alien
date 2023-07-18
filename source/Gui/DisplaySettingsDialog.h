#pragma once

#include "AlienDialog.h"
#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _DisplaySettingsDialog : public _AlienDialog
{
public:
    _DisplaySettingsDialog();

private:
    void processIntern();
    void openIntern();

    void setFullscreen(int selectionIndex);
    int getSelectionIndex() const;
    std::vector<std::string> createVideoModeStrings() const;

    std::string _origMode;
    int _origSelectionIndex;
    int _selectionIndex;
    int _origFps;

    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes;
    std::vector<std::string> _videoModeStrings;
};