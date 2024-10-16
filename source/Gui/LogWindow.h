#pragma once

#include "Definitions.h"
#include "AlienWindow.h"

class _LogWindow : public AlienWindow
{
public:
    _LogWindow(GuiLogger const& logger);
    ~_LogWindow();

private:
    void processIntern();

    bool _verbose = false;

    GuiLogger _logger;
};
