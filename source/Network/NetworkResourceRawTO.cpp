#include "NetworkResourceRawTO.h"

#include <ranges>
#include <imgui.h>

int _NetworkResourceRawTO::compare(NetworkResourceRawTO const& left, NetworkResourceRawTO const& right, ImGuiTableSortSpecs const* specs)
{
    for (int n = 0; n < specs->SpecsCount; n++) {
        const ImGuiTableColumnSortSpecs* sortSpec = &specs->Specs[n];
        int delta = 0;
        switch (sortSpec->ColumnUserID) {
        case NetworkResourceColumnId_Timestamp:
            delta = left->timestamp.compare(right->timestamp);
            break;
        case NetworkResourceColumnId_UserName:
            delta = left->userName.compare(right->userName);
            break;
        case NetworkResourceColumnId_SimulationName:
            delta = left->resourceName.compare(right->resourceName);
            break;
        case NetworkResourceColumnId_Description:
            delta = left->description.compare(right->description);
            break;
        case NetworkResourceColumnId_Likes:
            delta = left->getTotalLikes() - right->getTotalLikes();
            break;
        case NetworkResourceColumnId_NumDownloads:
            delta = left->numDownloads - right->numDownloads;
            break;
        case NetworkResourceColumnId_Width:
            delta = left->width - right->width;
            break;
        case NetworkResourceColumnId_Height:
            delta = left->height - right->height;
            break;
        case NetworkResourceColumnId_Particles:
            delta = left->particles - right->particles;
            break;
        case NetworkResourceColumnId_FileSize:
            delta = static_cast<int>(left->contentSize / 1024) - static_cast<int>(right->contentSize / 1024);
            break;
        case NetworkResourceColumnId_Version:
            delta = left->version.compare(right->version);
            break;
        }
        if (delta > 0) {
            return (sortSpec->SortDirection == ImGuiSortDirection_Ascending) ? +1 : -1;
        }
        if (delta < 0) {
            return (sortSpec->SortDirection == ImGuiSortDirection_Ascending) ? -1 : +1;
        }
    }

    return 0;
}

bool _NetworkResourceRawTO::matchWithFilter(std::string const& filter) const
{
    auto match = false;
    if (timestamp.find(filter) != std::string::npos) {
        match = true;
    }
    if (userName.find(filter) != std::string::npos) {
        match = true;
    }
    if (resourceName.find(filter) != std::string::npos) {
        match = true;
    }
    if (std::to_string(numDownloads).find(filter) != std::string::npos) {
        match = true;
    }
    if (std::to_string(width).find(filter) != std::string::npos) {
        match = true;
    }
    if (std::to_string(height).find(filter) != std::string::npos) {
        match = true;
    }
    if (std::to_string(particles).find(filter) != std::string::npos) {
        match = true;
    }
    if (std::to_string(contentSize).find(filter) != std::string::npos) {
        match = true;
    }
    if (description.find(filter) != std::string::npos) {
        match = true;
    }
    if (version.find(filter) != std::string::npos) {
        match = true;
    }
    return match;
}

int _NetworkResourceRawTO::getTotalLikes() const
{
    int result = 0;
    for (auto const& numReactions : numLikesByEmojiType | std::views::values) {
        result += numReactions;
    }
    return result;
}
