#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>

#include "AlienDialog.h"
#include "Definitions.h"

class DisplaySettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DisplaySettingsDialog);

public:
    ~DisplaySettingsDialog() override;

private:
    DisplaySettingsDialog();

    void processIntern() override;
    void openIntern() override;

    void setFullscreen(int selectionIndex);
    bool isContentScalingChanged() const;
    void onChangeContentScaling(bool automatic, float scaleFactor) const;
    int getSelectionIndex() const;
    std::vector<std::string> createVideoModeStrings() const;

    int _origSelectionIndex = 0;
    int _selectionIndex = 0;
    int _origFps = 0;
    bool _origAutoContentScaleFactor = true;
    float _origContentScaleFactor = 1.0f;

    bool _pendingIsFullscreen = false;
    int _pendingSelectionIndex = 0;
    int _pendingFps = 0;
    bool _pendingAutoContentScaleFactor = true;
    float _pendingContentScaleFactor = 1.0f;

    // Copies, because GLFW frees its mode array when the monitor reconnects (e.g. after standby)
    std::vector<GLFWvidmode> _videoModes;
    std::vector<std::string> _videoModeStrings;
};
