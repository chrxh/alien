#pragma once

#include <Network/Definitions.h>

#include <PersisterInterface/PersisterFacade.h>

#include "AlienDialog.h"
#include "Definitions.h"

class LoginDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(LoginDialog);

private:
    LoginDialog();

    void initIntern() override;
    void processIntern() override;

};
