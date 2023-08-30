#pragma once

#include "AlienDialog.h"
#include "Definitions.h"
#include "NetworkController.h"

class _NewPasswordDialog : public _AlienDialog
{
public:
    _NewPasswordDialog(SimulationController const& simController, BrowserWindow const& browserWindow, NetworkController const& networkController);

    void open(std::string const& userName, UserInfo const& userInfo);

private:
    void processIntern();

    void onNewPassword();

    SimulationController _simController; 
    BrowserWindow _browserWindow;
    NetworkController _networkController;

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
    UserInfo _userInfo;
};