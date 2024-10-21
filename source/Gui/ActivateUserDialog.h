#pragma once

#include "Network/NetworkService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ActivateUserDialog : public AlienDialog<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ActivateUserDialog);

public:
    void open(std::string const& userName, std::string const& password, UserInfo const& userInfo);

private:
    ActivateUserDialog();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;
    void onActivateUser();

    SimulationFacade _simulationFacade; 

    std::string _userName;
    std::string _password;
    std::string _confirmationCode;
    UserInfo _userInfo;
};