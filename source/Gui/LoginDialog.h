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
        CreateUserDialog const& createUserDialog,
        ActivateUserDialog const& activateUserDialog,
        ResetPasswordDialog const& resetPasswordDialog);
    ~_LoginDialog();

private:
    void processIntern();

    SimulationController _simController;
    PersisterController _persisterController; 
    BrowserWindow _browserWindow;
    CreateUserDialog _createUserDialog;
    ActivateUserDialog _activateUserDialog;
    ResetPasswordDialog _resetPasswordDialog;
};
