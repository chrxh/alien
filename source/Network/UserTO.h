#pragma once

#include <string>

class UserTO
{
public:
    std::string userName;
    int starsReceived;
    int starsGiven;
    std::string timestamp;
    bool online;
    bool lastDayOnline;
    int timeSpent;
    std::string gpu;

    static int compareOnlineAndTimestamp(UserTO const& left, UserTO const& right)
    {
        if (int result = static_cast<int>(left.online) - static_cast<int>(right.online)) {
            return result;
        }
        if (int result = left.timestamp.compare(right.timestamp)) {
            return result;
        }
        return 0;
    }
};

