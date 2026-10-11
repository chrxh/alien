#pragma once

#include <Network/Interface/NetworkService.h>

#include <Engine/Interface/SimulationFacade.h>

#include "AlienDialog.h"
#include "Definitions.h"

class NewPasswordDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NewPasswordDialog);

public:
    void open(std::string const& userName, UserInfo const& userInfo);

private:
    NewPasswordDialog();

    void initIntern() override;
    void processIntern() override;

    void onNewPassword();

    std::string _userName;
    std::string _newPassword;
    std::string _confirmationCode;
    UserInfo _userInfo;
};
