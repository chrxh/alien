#pragma once

#include <string>
#include <set>

struct GetUserNamesForReactionResultData
{
    std::string resourceId;
    int emojiType = 0;
    std::set<std::string> userNames;
};
