#pragma once

#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ResetPasswordDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ResetPasswordDialog);

public:
    void open(std::string const& userName, UserInfo const& userInfo);

private:
    ResetPasswordDialog();

    void processIntern() override;

    void onResetPassword();

    std::string _userName;
    std::string _email;
    UserInfo _userInfo;
};