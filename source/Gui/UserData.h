#pragma once
#include <string>

class UserData
{
public:
    std::string userName;
    int starsReceived;
    int starsGiven;
    std::string timestamp;
    bool online;

    static int compare(UserData const& left, UserData const& right) { return left.timestamp.compare(right.timestamp); }
};

