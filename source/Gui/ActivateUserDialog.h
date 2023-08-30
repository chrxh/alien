#pragma once

#include "AlienDialog.h"
#include "Definitions.h"
#include "NetworkController.h"

class _ActivateUserDialog : public _AlienDialog
{
public:
    _ActivateUserDialog(SimulationController const& simController, BrowserWindow const& browserWindow, NetworkController const& networkController);
    ~_ActivateUserDialog();

    void registerCyclicReferences(CreateUserDialogWeakPtr const& createUserDialog);

    void open(std::string const& userName, std::string const& password, UserInfo const& userInfo);

private:
    void processIntern() override;
    void onActivateUser();

    SimulationController _simController; 
    BrowserWindow _browserWindow;
    NetworkController _networkController;
    CreateUserDialogWeakPtr _createUserDialog;

    std::string _userName;
    std::string _password;
    std::string _confirmationCode;
    UserInfo _userInfo;
};