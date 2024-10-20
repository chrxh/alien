#pragma once

#include <string>

struct GetUserNamesForReactionRequestData
{
    std::string resourceId;
    int emojiType = 0;
};
