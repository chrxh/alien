#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienDialog.h"

class DisplaySettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DisplaySettingsDialog);

public:
    void init();

private:
    DisplaySettingsDialog();

    void processIntern();
    void openIntern();

    void setFullscreen(int selectionIndex);
    int getSelectionIndex() const;
    std::vector<std::string> createVideoModeStrings() const;

    std::string _origMode;
    int _origSelectionIndex = 0;
    int _selectionIndex = 0;
    int _origFps = 33;

    int _videoModesCount = 0;
    GLFWvidmode const* _videoModes;
    std::vector<std::string> _videoModeStrings;
};