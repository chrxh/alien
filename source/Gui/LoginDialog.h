#pragma once

#include "Definitions.h"

class _LoginDialog
{
public:
    _LoginDialog(BrowserWindow const& browserWindow, CreateUserDialog const& createUserDialog, NetworkController const& networkController);
    ~_LoginDialog();

    void process();

    void show();

private:
    void onLogin();

    BrowserWindow _browserWindow;
    CreateUserDialog _createUserDialog;
    NetworkController _networkController;

    bool _show = false;
    bool _remember = true;
    std::string _userName;
    std::string _password;
};