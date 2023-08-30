#pragma once

#include "AlienDialog.h"
#include "Definitions.h"
#include "NetworkController.h"

class _ResetPasswordDialog : public _AlienDialog
{
public:
    _ResetPasswordDialog(NewPasswordDialog const& newPasswordDialog, NetworkController const& networkController);

    void open(std::string const& userName, UserInfo const& userInfo);

private:
    void processIntern();

    void onResetPassword();

    NewPasswordDialog _newPasswordDialog; 
    NetworkController _networkController;

    std::string _userName;
    std::string _email;
    UserInfo _userInfo;
};