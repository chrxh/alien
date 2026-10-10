#pragma once
#include <Network/Interface/NetworkService.h>

struct LoginRequestData
{
    std::string userName;
    std::string password;
    UserInfo userInfo;
};
