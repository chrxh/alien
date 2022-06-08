#pragma once

#include "Definitions.h"

class _DeleteUserDialog
{
public:
    _DeleteUserDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);
    ~_DeleteUserDialog();

    void process();

    void show();

private:
    void onDelete();

    BrowserWindow _browserWindow;
    NetworkController _networkController;

    bool _show = false;
    std::string _reenteredPassword;
};
