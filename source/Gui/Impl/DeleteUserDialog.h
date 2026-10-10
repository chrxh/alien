#pragma once

#include <Base/Interface/Singleton.h>

#include <Network/Interface/Definitions.h>

#include "AlienDialog.h"
#include "Definitions.h"

class DeleteUserDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DeleteUserDialog);

private:
    DeleteUserDialog();

    void processIntern();
    void onDelete();

    std::string _reenteredPassword;
};
