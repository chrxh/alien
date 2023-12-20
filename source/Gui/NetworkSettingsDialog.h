#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _NetworkSettingsDialog : public _AlienDialog
{
public:
    _NetworkSettingsDialog(BrowserWindow const& browserWindow, NetworkService const& networkController);

private:
    void processIntern();
    void openIntern();

    void onChangeSettings();

    BrowserWindow _browserWindow;
    NetworkService _networkService;

    std::string _serverAddress;
    std::string _origServerAddress;
};