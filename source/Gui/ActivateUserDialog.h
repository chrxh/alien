#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _ActivateUserDialog : public _AlienDialog
{
public:
    _ActivateUserDialog(BrowserWindow const& browserWindow, NetworkController const& networkController);
    ~_ActivateUserDialog();

    void registerCyclicReferences(CreateUserDialogWeakPtr const& createUserDialog);

    void open(std::string const& userName, std::string const& password);

private:
    void processIntern() override;
    void onActivateUser();

    BrowserWindow _browserWindow;
    NetworkController _networkController;
    CreateUserDialogWeakPtr _createUserDialog;

    std::string _userName;
    std::string _password;
    std::string _confirmationCode;
};