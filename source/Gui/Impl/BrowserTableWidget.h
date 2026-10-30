#pragma once

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <imgui.h>

#include <Network/NetworkResourceTreeTO.h>

#include "Definitions.h"

class _BrowserTableWidget
{
public:
    static BrowserTableWidget create(BrowserData const& data);

    void process();

private:
    _BrowserTableWidget(BrowserData const& data);

    struct Column
    {
        std::string name;
        ImGuiTableColumnFlags flags = ImGuiTableColumnFlags_WidthFixed;
        float width = 0;  // Unscaled
        int sortId = 0;
        std::function<void(NetworkResourceTreeTO const&)> processField;
    };
    std::vector<Column> getColumns();

    void processSortSpecs();
    void processRow(NetworkResourceTreeTO const& treeTO, std::vector<Column> const& columns);

    void processResourceNameField(NetworkResourceTreeTO const& treeTO);
    void processFolderTreeSymbols(NetworkResourceTreeTO const& treeTO);
    void processDescriptionField(NetworkResourceTreeTO const& treeTO);
    void processReactionList(NetworkResourceTreeTO const& treeTO);
    void processTimestampField(NetworkResourceTreeTO const& treeTO);
    void processUserNameField(NetworkResourceTreeTO const& treeTO);
    void processNumDownloadsField(NetworkResourceTreeTO const& treeTO);
    void processWidthField(NetworkResourceTreeTO const& treeTO);
    void processHeightField(NetworkResourceTreeTO const& treeTO);
    void processNumObjectsField(NetworkResourceTreeTO const& treeTO, bool kobjects);
    void processSizeField(NetworkResourceTreeTO const& treeTO, bool kbyte);
    void processVersionField(NetworkResourceTreeTO const& treeTO);

    BrowserData _data;
    std::optional<std::set<std::vector<std::string>>> _pendingCollapsedFolderNames;
};
