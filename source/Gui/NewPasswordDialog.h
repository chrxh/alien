#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _NewPasswordDialog : public _AlienDialog
{
public:
    _NewPasswordDialog(SimulationFacade const& simulationFacade, BrowserWindow const& browserWindow);

    void open(std::string const& userName, UserInfo const& userInfo);

private:
    void processIntern();

    void onNewPassword();

    SimulationFacade _simulationFacade; 
    BrowserWindow _browserWindow;

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
    UserInfo _userInfo;
};