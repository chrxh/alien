#pragma once

#include "Definitions.h"

class _ResetPasswordDialog
{
public:
    _ResetPasswordDialog(NewPasswordDialog const& newPasswordDialog, NetworkController const& networkController);
    ~_ResetPasswordDialog();

    void process();

    void show(std::string const& userName);

private:
    void onResetPassword();

    NewPasswordDialog _newPasswordDialog; 
    NetworkController _networkController;

    bool _show = false;
    std::string _userName;
    std::string _email;
};