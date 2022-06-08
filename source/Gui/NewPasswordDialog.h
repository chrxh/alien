#pragma once

#include "Definitions.h"

class _NewPasswordDialog
{
public:
    _NewPasswordDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);
    ~_NewPasswordDialog();

    void process();

    void show(std::string const& userName);

private:
    void onNewPassword();

    BrowserWindow _browserWindow;
    NetworkController _networkController;

    bool _show = false;
    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
};