#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _NewPasswordDialog : public _AlienDialog
{
public:
    _NewPasswordDialog(SimulationController const& simController, BrowserWindow const& browserWindow, NetworkService const& networkController);

    void open(std::string const& userName, UserInfo const& userInfo);

private:
    void processIntern();

    void onNewPassword();

    SimulationController _simController; 
    BrowserWindow _browserWindow;
    NetworkService _networkService;

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
    UserInfo _userInfo;
};