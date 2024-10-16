#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _ActivateUserDialog : public _AlienDialog
{
public:
    _ActivateUserDialog(SimulationFacade const& simulationFacade, BrowserWindow const& browserWindow);
    ~_ActivateUserDialog();

    void registerCyclicReferences(CreateUserDialogWeakPtr const& createUserDialog);

    void open(std::string const& userName, std::string const& password, UserInfo const& userInfo);

private:
    void processIntern() override;
    void onActivateUser();

    SimulationFacade _simulationFacade; 
    BrowserWindow _browserWindow;
    CreateUserDialogWeakPtr _createUserDialog;

    std::string _userName;
    std::string _password;
    std::string _confirmationCode;
    UserInfo _userInfo;
};