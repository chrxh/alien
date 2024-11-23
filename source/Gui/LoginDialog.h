#pragma once

#include "Network/Definitions.h"
#include "PersisterInterface/PersisterFacade.h"

#include "AlienDialog.h"
#include "Definitions.h"

class LoginDialog : public AlienDialog<SimulationFacade, PersisterFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LoginDialog);

private:
    LoginDialog();

    void initIntern(SimulationFacade simulationFacade, PersisterFacade persisterFacade) override;
    void processIntern() override;

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade; 
};
