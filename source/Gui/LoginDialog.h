#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _LoginDialog : public _AlienDialog
{
public:
    _LoginDialog(
        SimulationController const& simController, 
        BrowserWindow const& browserWindow,
        CreateUserDialog const& createUserDialog,
        ActivateUserDialog const& activateUserDialog,
        ResetPasswordDialog const& resetPasswordDialog,
        NetworkController const& networkController);
    ~_LoginDialog();

private:
    void processIntern();

    void onLogin();

    UserInfo getUserInfo();

    SimulationController _simController; 
    BrowserWindow _browserWindow;
    CreateUserDialog _createUserDialog;
    ActivateUserDialog _activateUserDialog;
    NetworkController _networkController;
    ResetPasswordDialog _resetPasswordDialog;

    bool _shareGpuInfo = true;
    bool _remember = true;
    std::string _userName;
    std::string _password;
};