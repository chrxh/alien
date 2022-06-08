#pragma once

#include "Definitions.h"

class _NetworkSettingsDialog
{
public:
    _NetworkSettingsDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);

    ~_NetworkSettingsDialog();

    void process();

    void show();

private:
    void onChangeSettings();

    BrowserWindow _browserWindow;
    NetworkController _networkController;

    bool _show = false;
    std::string _serverAddress;
    std::string _origServerAddress;
};