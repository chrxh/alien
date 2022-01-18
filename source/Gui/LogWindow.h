#pragma once

#include "Definitions.h"
#include "AlienWindow.h"

class _LogWindow : public _AlienWindow
{
public:
    _LogWindow(SimpleLogger const& logger);
    ~_LogWindow();

private:
    void processIntern();

    bool _verbose = false;

    SimpleLogger _logger;
};
