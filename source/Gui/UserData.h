#pragma once
#include <string>

class UserData
{
public:
    std::string userName;
    int starsEarned;
    int starsGiven;
    std::string timestamp;

    static int compare(UserData const& left, UserData const& right) { return left.timestamp.compare(right.timestamp); }
};

