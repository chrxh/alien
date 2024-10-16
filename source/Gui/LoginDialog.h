#pragma once

#include "Network/Definitions.h"
#include "PersisterInterface/PersisterFacade.h"

#include "AlienDialog.h"
#include "Definitions.h"

class LoginDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTOR(LoginDialog);

public:
    void init(
        SimulationFacade const& simulationFacade,
        PersisterFacade const& persisterFacade,
        CreateUserDialog const& createUserDialog,
        ActivateUserDialog const& activateUserDialog,
        ResetPasswordDialog const& resetPasswordDialog);

private:
    LoginDialog();

    void processIntern();

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade; 
    CreateUserDialog _createUserDialog;
    ActivateUserDialog _activateUserDialog;
    ResetPasswordDialog _resetPasswordDialog;
};
