#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "AlienWindow.h"

class LogWindow : public AlienWindow<GuiLogger>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LogWindow);

private:
    LogWindow();

    void initIntern(GuiLogger logger) override;
    void shutdownIntern() override;
    void processIntern() override;

    bool _verbose = false;

    GuiLogger _logger;
};
