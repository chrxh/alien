#pragma once

#include <Network/Interface/Definitions.h>

#include <Persister/Interface/PersisterFacade.h>

#include "AlienDialog.h"
#include "Definitions.h"

class LoginDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LoginDialog);

private:
    LoginDialog();

    void initIntern() override;
    void processIntern() override;

};
