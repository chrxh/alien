#pragma once

#include "Network/Definitions.h"
#include "PersisterInterface/PersisterController.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _LoginDialog : public _AlienDialog
{
public:
    _LoginDialog(
        SimulationController const& simController,
        PersisterController const& persisterController,
        BrowserWindow const& browserWindow,
        CreateUserDialog const& createUserDialog,
        ActivateUserDialog const& activateUserDialog,
        ResetPasswordDialog const& resetPasswordDialog);
    ~_LoginDialog();

private:
    void processIntern();

    void onLogin();

    SimulationController _simController;
    PersisterController _persisterController; 
    BrowserWindow _browserWindow;
    CreateUserDialog _createUserDialog;
    ActivateUserDialog _activateUserDialog;
    ResetPasswordDialog _resetPasswordDialog;

    std::string _userName;
    std::string _password;
};
