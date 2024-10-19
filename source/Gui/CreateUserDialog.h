#pragma once

#include "Base/Singleton.h"
#include "Network/NetworkService.h"

#include "AlienDialog.h"
#include "Definitions.h"

class CreateUserDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreateUserDialog);

public:
    void open(std::string const& userName, std::string const& password, UserInfo const& userInfo);

    void onCreateUser();
private:
    CreateUserDialog();

    void processIntern();

    std::string _userName;
    std::string _password;
    std::string _email;
    UserInfo _userInfo;
};