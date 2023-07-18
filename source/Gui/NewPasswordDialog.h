#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _NewPasswordDialog : public _AlienDialog
{
public:
    _NewPasswordDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);

    void open(std::string const& userName);

private:
    void processIntern();

    void onNewPassword();

    BrowserWindow _browserWindow;
    NetworkController _networkController;

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
};