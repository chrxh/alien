#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _CreateUserDialog : public AlienDialog
{
public:
    _CreateUserDialog(ActivateUserDialog const& activateUserDialog);
    ~_CreateUserDialog();

    void open(std::string const& userName, std::string const& password, UserInfo const& userInfo);

    void onCreateUser();
private:
    void processIntern();

    ActivateUserDialog _activateUserDialog; 

    std::string _userName;
    std::string _password;
    std::string _email;
    UserInfo _userInfo;
};