#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _DeleteUserDialog : public AlienDialog
{
public:
    _DeleteUserDialog();

private:
    void processIntern();
    void onDelete();

    std::string _reenteredPassword;
};
