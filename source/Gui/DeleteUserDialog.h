#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _DeleteUserDialog : public _AlienDialog
{
public:
    _DeleteUserDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);

private:
    void processIntern();
    void onDelete();

    BrowserWindow _browserWindow;
    NetworkController _networkController;

    std::string _reenteredPassword;
};
