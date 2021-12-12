#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _DisplaySettingsDialog
{
public:
    _DisplaySettingsDialog(WindowController const& windowController);
    ~_DisplaySettingsDialog();

    void process();

    void show();

private:

    WindowController _windowController;
    bool _show = false;
    bool _origFullscreen;  
};