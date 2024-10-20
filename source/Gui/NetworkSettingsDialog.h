#pragma once

#include "Base/Singleton.h"
#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class NetworkSettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NetworkSettingsDialog);

private:
    NetworkSettingsDialog();

    void processIntern();
    void openIntern();

    void onChangeSettings();

    std::string _serverAddress;
    std::string _origServerAddress;
};