#pragma once

#include "Network/Definitions.h"
#include "PersisterInterface/PersisterFacade.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _LoginDialog : public _AlienDialog
{
public:
    _LoginDialog(
        SimulationFacade const& simulationFacade,
        PersisterFacade const& persisterFacade,
        CreateUserDialog const& createUserDialog,
        ActivateUserDialog const& activateUserDialog,
        ResetPasswordDialog const& resetPasswordDialog);
    ~_LoginDialog();

private:
    void processIntern();

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade; 
    CreateUserDialog _createUserDialog;
    ActivateUserDialog _activateUserDialog;
    ResetPasswordDialog _resetPasswordDialog;
};
