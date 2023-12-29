#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _DeleteUserDialog : public _AlienDialog
{
public:
    _DeleteUserDialog(BrowserWindow const& browserWindow);

private:
    void processIntern();
    void onDelete();

    BrowserWindow _browserWindow;

    std::string _reenteredPassword;
};
