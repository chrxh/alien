#pragma once

struct MoveNetworkResourceRequestData
{
    struct Entry
    {
        std::string resourceId;
        WorkspaceType workspaceType;
    };
    std::vector<Entry> entries;
};
