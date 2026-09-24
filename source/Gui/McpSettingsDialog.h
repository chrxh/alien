#pragma once

#include <Base/Singleton.h>

#include "AlienDialog.h"
#include "Definitions.h"

class McpSettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(McpSettingsDialog);

private:
    McpSettingsDialog();

    void processIntern() override;
    void openIntern() override;

    int _port = 0;
};
