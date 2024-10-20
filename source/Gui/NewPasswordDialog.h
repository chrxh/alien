#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class NewPasswordDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NewPasswordDialog);
public:
    void init(SimulationFacade const& simulationFacade);

    void open(std::string const& userName, UserInfo const& userInfo);

private:
    NewPasswordDialog();

    void processIntern();

    void onNewPassword();

    SimulationFacade _simulationFacade; 

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
    UserInfo _userInfo;
};