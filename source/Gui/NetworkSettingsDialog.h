#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _NetworkSettingsDialog : public _AlienDialog
{
public:
    _NetworkSettingsDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);

private:
    void processIntern();
    void openIntern();

    void onChangeSettings();

    BrowserWindow _browserWindow;
    NetworkController _networkController;

    std::string _serverAddress;
    std::string _origServerAddress;
};