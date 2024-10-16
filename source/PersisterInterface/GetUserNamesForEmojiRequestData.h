#pragma once

#include <string>

struct GetUserNamesForEmojiRequestData
{
    std::string resourceId;
    int emojiType = 0;
};
