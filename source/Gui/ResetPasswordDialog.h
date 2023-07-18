#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _ResetPasswordDialog : public _AlienDialog
{
public:
    _ResetPasswordDialog(NewPasswordDialog const& newPasswordDialog, NetworkController const& networkController);

    void open(std::string const& userName);

private:
    void processIntern();

    void onResetPassword();

    NewPasswordDialog _newPasswordDialog; 
    NetworkController _networkController;

    std::string _userName;
    std::string _email;
};