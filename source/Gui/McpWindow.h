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
    void processServerSettings();
    void processConnectionInfo();
    void processCommandLog();

    void processCopyableText(std::string const& id, std::string text, int numLines = 1);
};
