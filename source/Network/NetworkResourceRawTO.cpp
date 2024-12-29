#include "NetworkResourceRawTO.h"

#include <algorithm>
#include <ranges>
#include <imgui.h>

#include "Base/StringHelper.h"

int _NetworkResourceRawTO::compare(NetworkResourceRawTO const& left, NetworkResourceRawTO const& right, std::vector<ImGuiTableColumnSortSpecs> const& sortSpecs)
{
    for (auto const& sortSpec : sortSpecs) {
        int delta = 0;
        switch (sortSpec.ColumnUserID) {
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
        default:
            break;
        }
        if (delta > 0) {
            return (sortSpec.SortDirection == ImGuiSortDirection_Ascending) ? +1 : -1;
        }
        if (delta < 0) {
            return (sortSpec.SortDirection == ImGuiSortDirection_Ascending) ? -1 : +1;
        }
    }
    return 0;
}

bool _NetworkResourceRawTO::matchWithFilter(std::string const& filter) const
{
    auto match = false;
    if (StringHelper::containsCaseInsensitive(timestamp, filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(userName, filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(resourceName, filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(std::to_string(numDownloads), filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(std::to_string(width), filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(std::to_string(height), filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(std::to_string(particles), filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(std::to_string(contentSize), filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(description, filter)) {
        match = true;
    }
    if (StringHelper::containsCaseInsensitive(version, filter)) {
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
