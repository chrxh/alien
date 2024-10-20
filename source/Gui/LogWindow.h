#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "AlienWindow.h"

class LogWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LogWindow);

public:
    void init(GuiLogger const& logger);
    void shutdown();

private:
    LogWindow();

    void processIntern();

    bool _verbose = false;

    GuiLogger _logger;
};
