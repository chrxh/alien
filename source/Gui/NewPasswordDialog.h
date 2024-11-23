#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"
#include "EngineInterface/SimulationFacade.h"

class NewPasswordDialog : public AlienDialog<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NewPasswordDialog);
public:
    void open(std::string const& userName, UserInfo const& userInfo);

private:
    NewPasswordDialog();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;

    void onNewPassword();

    SimulationFacade _simulationFacade; 

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
    UserInfo _userInfo;
};