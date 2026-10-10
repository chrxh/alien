#pragma once

#include <set>
#include <string>

struct GetUserNamesForReactionResultData
{
    std::string resourceId;
    int emojiType = 0;
    std::set<std::string> userNames;
};
