#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "AlienWindow.h"

class LogWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LogWindow);

public:
    void init(GuiLogger const& logger);

private:
    LogWindow();

    void shutdownIntern() override;
    void processIntern() override;

    bool _verbose = false;

    GuiLogger _logger;
};
