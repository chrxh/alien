#pragma once

#include "AlienDialog.h"
#include "Definitions.h"
#include "NetworkController.h"

class _CreateUserDialog : public _AlienDialog
{
public:
    _CreateUserDialog(ActivateUserDialog const& activateUserDialog, NetworkController const& networkController);
    ~_CreateUserDialog();

    void open(std::string const& userName, std::string const& password, UserInfo const& userInfo);

    void onCreateUser();
private:
    void processIntern();

    NetworkController _networkController;
    ActivateUserDialog _activateUserDialog; 

    std::string _userName;
    std::string _password;
    std::string _email;
    UserInfo _userInfo;
};