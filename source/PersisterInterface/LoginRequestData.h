#pragma once
#include "Network/NetworkService.h"

struct LoginRequestData
{
    std::string userName;
    std::string password;
    UserInfo userInfo;
};
