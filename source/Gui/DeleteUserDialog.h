#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _DeleteUserDialog : public _AlienDialog
{
public:
    _DeleteUserDialog(BrowserWindow const& browserWindow, NetworkService const& networkController);

private:
    void processIntern();
    void onDelete();

    BrowserWindow _browserWindow;
    NetworkService _networkService;

    std::string _reenteredPassword;
};
