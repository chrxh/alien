#pragma once

#include "Network/Definitions.h"
#include "PersisterInterface/PersisterFacade.h"

#include "AlienDialog.h"
#include "Definitions.h"

class LoginDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LoginDialog);

public:
    void init(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade);

private:
    LoginDialog();

    void processIntern();

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade; 
};
