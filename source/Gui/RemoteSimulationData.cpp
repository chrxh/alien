#include "RemoteSimulationData.h"

#include <ranges>
#include <imgui.h>

int _RemoteSimulationData::compare(RemoteSimulationData const& left, RemoteSimulationData const& right, ImGuiTableSortSpecs const* specs)
{
    for (int n = 0; n < specs->SpecsCount; n++) {
        const ImGuiTableColumnSortSpecs* sortSpec = &specs->Specs[n];
        int delta = 0;
        switch (sortSpec->ColumnUserID) {
        case RemoteSimulationDataColumnId_Timestamp:
            delta = left->timestamp.compare(right->timestamp);
            break;
        case RemoteSimulationDataColumnId_UserName:
            delta = left->userName.compare(right->userName);
            break;
        case RemoteSimulationDataColumnId_SimulationName:
            delta = left->simName.compare(right->simName);
            break;
        case RemoteSimulationDataColumnId_Description:
            delta = left->description.compare(right->description);
            break;
        case RemoteSimulationDataColumnId_Likes:
            delta = left->getTotalLikes() - right->getTotalLikes();
            break;
        case RemoteSimulationDataColumnId_NumDownloads:
            delta = left->numDownloads - right->numDownloads;
            break;
        case RemoteSimulationDataColumnId_Width:
            delta = left->width - right->width;
            break;
        case RemoteSimulationDataColumnId_Height:
            delta = left->height - right->height;
            break;
        case RemoteSimulationDataColumnId_Particles:
            delta = left->particles - right->particles;
            break;
        case RemoteSimulationDataColumnId_FileSize:
            delta = static_cast<int>(left->contentSize / 1024) - static_cast<int>(right->contentSize / 1024);
            break;
        case RemoteSimulationDataColumnId_Version:
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

bool _RemoteSimulationData::matchWithFilter(std::string const& filter) const
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

int _RemoteSimulationData::getTotalLikes() const
{
    int result = 0;
    for (auto const& numReactions : numLikesByEmojiType | std::views::values) {
        result += numReactions;
    }
    return result;
}
