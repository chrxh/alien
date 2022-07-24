#include "RemoteSimulationData.h"

#include <imgui.h>

int RemoteSimulationData::compare(void const* left, void const* right, ImGuiTableSortSpecs const* specs)
{
    auto leftImpl = reinterpret_cast<RemoteSimulationData const*>(left);
    auto rightImpl = reinterpret_cast<RemoteSimulationData const*>(right);
    for (int n = 0; n < specs->SpecsCount; n++) {
        // Here we identify columns using the ColumnUserID value that we ourselves passed to TableSetupColumn()
        // We could also choose to identify columns based on their index (sort_spec->ColumnIndex), which is simpler!
        const ImGuiTableColumnSortSpecs* sortSpec = &specs->Specs[n];
        int delta = 0;
        switch (sortSpec->ColumnUserID) {
        case RemoteSimulationDataColumnId_Timestamp:
            delta = leftImpl->timestamp.compare(rightImpl->timestamp);
            break;
        case RemoteSimulationDataColumnId_UserName:
            delta = leftImpl->userName.compare(rightImpl->userName);
            break;
        case RemoteSimulationDataColumnId_SimulationName:
            delta = leftImpl->simName.compare(rightImpl->simName);
            break;
        case RemoteSimulationDataColumnId_Description:
            delta = leftImpl->description.compare(rightImpl->description);
            break;
        case RemoteSimulationDataColumnId_Likes:
            delta = (leftImpl->likes - rightImpl->likes);
            break;
        case RemoteSimulationDataColumnId_NumDownloads:
            delta = (leftImpl->numDownloads - rightImpl->numDownloads);
            break;
        case RemoteSimulationDataColumnId_Width:
            delta = (leftImpl->width - rightImpl->width);
            break;
        case RemoteSimulationDataColumnId_Height:
            delta = (leftImpl->height - rightImpl->height);
            break;
        case RemoteSimulationDataColumnId_Particles:
            delta = (leftImpl->particles - rightImpl->particles);
            break;
        case RemoteSimulationDataColumnId_FileSize:
            delta = static_cast<int>(leftImpl->contentSize / 1024) - static_cast<int>(rightImpl->contentSize / 1024);
            break;
        case RemoteSimulationDataColumnId_Version:
            delta = leftImpl->version.compare(rightImpl->version);
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

bool RemoteSimulationData::matchWithFilter(std::string const& filter) const
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
    if (std::to_string(likes).find(filter) != std::string::npos) {
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
