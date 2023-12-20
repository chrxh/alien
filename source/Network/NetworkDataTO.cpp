#include "NetworkDataTO.h"

#include <ranges>
#include <imgui.h>

int _NetworkDataTO::compare(NetworkDataTO const& left, NetworkDataTO const& right, ImGuiTableSortSpecs const* specs)
{
    for (int n = 0; n < specs->SpecsCount; n++) {
        const ImGuiTableColumnSortSpecs* sortSpec = &specs->Specs[n];
        int delta = 0;
        switch (sortSpec->ColumnUserID) {
        case NetworkDataColumnId_Timestamp:
            delta = left->timestamp.compare(right->timestamp);
            break;
        case NetworkDataColumnId_UserName:
            delta = left->userName.compare(right->userName);
            break;
        case NetworkDataColumnId_SimulationName:
            delta = left->simName.compare(right->simName);
            break;
        case NetworkDataColumnId_Description:
            delta = left->description.compare(right->description);
            break;
        case NetworkDataColumnId_Likes:
            delta = left->getTotalLikes() - right->getTotalLikes();
            break;
        case NetworkDataColumnId_NumDownloads:
            delta = left->numDownloads - right->numDownloads;
            break;
        case NetworkDataColumnId_Width:
            delta = left->width - right->width;
            break;
        case NetworkDataColumnId_Height:
            delta = left->height - right->height;
            break;
        case NetworkDataColumnId_Particles:
            delta = left->particles - right->particles;
            break;
        case NetworkDataColumnId_FileSize:
            delta = static_cast<int>(left->contentSize / 1024) - static_cast<int>(right->contentSize / 1024);
            break;
        case NetworkDataColumnId_Version:
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

bool _NetworkDataTO::matchWithFilter(std::string const& filter) const
{
    auto match = false;
    if (timestamp.find(filter) != std::string::npos) {
        match = true;
    }
    if (userName.find(filter) != std::string::npos) {
        match = true;
    }
    if (simName.find(filter) != std::string::npos) {
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

int _NetworkDataTO::getTotalLikes() const
{
    int result = 0;
    for (auto const& numReactions : numLikesByEmojiType | std::views::values) {
        result += numReactions;
    }
    return result;
}
