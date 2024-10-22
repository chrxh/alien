#pragma once
#include <string>

struct EditNetworkResourceRequestData
{
    struct Entry
    {
        std::string resourceId;
        std::string newName;
        std::string newDescription;
    };
    std::vector<Entry> entries;
};
