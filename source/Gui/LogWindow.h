#pragma once

#include <Base/Singleton.h>

#include "AlienWindow.h"
#include "Definitions.h"

class LogWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LogWindow);

private:
    LogWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;

    bool _verbose = false;

    GuiLogger _logger;
};
