#pragma once
#include <string>

struct DeleteNetworkResourceRequestData
{
    struct Entry
    {
        std::string resourceId;
    };
    std::vector<Entry> entries;
};
