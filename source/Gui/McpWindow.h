#pragma once

#include <string>

#include <Base/Singleton.h>

#include "AlienWindow.h"
#include "Definitions.h"

class McpWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(McpWindow);

private:
    McpWindow();

    void processIntern() override;

    void processToolbar();
    void processStatusLine();
    void processStatusBadge(bool running);
    void processConnectionCard();
    void processCommandLog();
};
