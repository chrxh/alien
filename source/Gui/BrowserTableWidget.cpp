#include "BrowserTableWidget.h"

#include <map>
#include <optional>
#include <ranges>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>
#include <Base/VersionParserService.h>

#include <Network/NetworkResourceService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserGui.h"
#include "StyleRepository.h"

BrowserTableWidget _BrowserTableWidget::create(BrowserData const& data)
{
    return BrowserTableWidget(new _BrowserTableWidget(data));
}

_BrowserTableWidget::_BrowserTableWidget(BrowserData const& data)
    : _data(data)
{}

namespace
{
    ImGuiTableFlags getTableFlags(NetworkResourceType resourceType)
    {
        auto result = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable | ImGuiTableFlags_SortMulti
            | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;
        if (resourceType == NetworkResourceType_Genome) {
            result |= ImGuiTableFlags_NoBordersInBody;
        }
        return result;
    }
}

void _BrowserTableWidget::process()
{
    auto resourceType = _data->currentWorkspace.resourceType;
    auto& workspace = _data->getCurrentWorkspace();
    auto columns = getColumns(workspace);

    ImGui::PushID(resourceType == NetworkResourceType_Simulation ? "SimulationList" : "GenomeList");
    if (ImGui::BeginTable("Browser", toInt(columns.size()), getTableFlags(resourceType), ImVec2(-1, -scale(BrowserGui::WorkspaceBottomSpace)), 0.0f)) {
        for (auto const& column : columns) {
            ImGui::TableSetupColumn(column.name.c_str(), column.flags, scale(column.width), column.sortId);
        }
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        processSortSpecs();

        _recreateTreeTOs = false;
        ImGuiListClipper clipper;
        clipper.Begin(toInt(workspace.treeTOs.size()));
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                ImGui::PushID(row);
                processRow(workspace.treeTOs.at(row), columns);
                ImGui::PopID();
            }
        }
        ImGui::EndTable();

        if (_recreateTreeTOs) {
            _data->createTreeTOs(workspace);
        }
    }
    ImGui::PopID();
}

std::vector<_BrowserTableWidget::Column> _BrowserTableWidget::getColumns(Workspace& workspace)
{
    auto isSimulation = _data->currentWorkspace.resourceType == NetworkResourceType_Simulation;

    std::vector<Column> result;
    result.emplace_back(Column{
        .name = isSimulation ? "Simulation" : "Genome",
        .width = 210.0f,
        .sortId = NetworkResourceColumnId_SimulationName,
        .processField = [&, this](auto const& treeTO) { processResourceNameField(treeTO, workspace.collapsedFolderNames); }});
    result.emplace_back(Column{.name = "Description", .width = 200.0f, .sortId = NetworkResourceColumnId_Desc, .processField = [this](auto const& treeTO) {
                                   processDescriptionField(treeTO);
                               }});
    result.emplace_back(Column{.name = "Reactions", .width = 140.0f, .sortId = NetworkResourceColumnId_Likes, .processField = [this](auto const& treeTO) {
                                   processReactionList(treeTO);
                               }});
    result.emplace_back(Column{
        .name = "Timestamp",
        .flags = ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
        .width = 135.0f,
        .sortId = NetworkResourceColumnId_Timestamp,
        .processField = [this](auto const& treeTO) { processTimestampField(treeTO); }});
    result.emplace_back(Column{.name = "User name", .width = 120.0f, .sortId = NetworkResourceColumnId_UserName, .processField = [this](auto const& treeTO) {
                                   processUserNameField(treeTO);
                               }});
    result.emplace_back(Column{
        .name = "Downloads", .sortId = NetworkResourceColumnId_NumDownloads, .processField = [this](auto const& treeTO) { processNumDownloadsField(treeTO); }});
    if (isSimulation) {
        result.emplace_back(
            Column{.name = "Width", .sortId = NetworkResourceColumnId_Width, .processField = [this](auto const& treeTO) { processWidthField(treeTO); }});
        result.emplace_back(
            Column{.name = "Height", .sortId = NetworkResourceColumnId_Height, .processField = [this](auto const& treeTO) { processHeightField(treeTO); }});
    }
    result.emplace_back(
        Column{.name = isSimulation ? "Objects" : "Cells", .sortId = NetworkResourceColumnId_Particles, .processField = [=, this](auto const& treeTO) {
                   processNumObjectsField(treeTO, isSimulation);
               }});
    result.emplace_back(Column{.name = "File size", .sortId = NetworkResourceColumnId_FileSize, .processField = [=, this](auto const& treeTO) {
                                   processSizeField(treeTO, isSimulation);
                               }});
    result.emplace_back(
        Column{.name = "Version", .sortId = NetworkResourceColumnId_Version, .processField = [this](auto const& treeTO) { processVersionField(treeTO); }});
    return result;
}

void _BrowserTableWidget::processSortSpecs()
{
    auto sortSpecs = ImGui::TableGetSortSpecs();
    if (sortSpecs == nullptr || !sortSpecs->SpecsDirty) {
        return;
    }
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        auto& workspace = _data->workspaces.at(WorkspaceId{_data->currentWorkspace.resourceType, workspaceType});
        workspace.sortSpecs.clear();
        for (int i = 0; i < sortSpecs->SpecsCount; ++i) {
            workspace.sortSpecs.emplace_back(sortSpecs->Specs[i]);
        }
        _data->createTreeTOs(workspace);
    }
    sortSpecs->SpecsDirty = false;
}

namespace
{
    void pushTextColor(NetworkResourceTreeTO const& treeTO)
    {
        if (treeTO->isLeaf()) {
            auto const& leaf = treeTO->getLeaf();
            if (VersionParserService::get().isVersionOutdated(leaf.rawTO->version)) {
                ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::TextDecentColor);
            } else if (VersionParserService::get().isVersionNewer(leaf.rawTO->version)) {
                ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::TextConflictColor);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserLeafTextColor);
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserResourceTextColor);
        }
    }

    void popTextColor()
    {
        ImGui::PopStyleColor();
    }
}

void _BrowserTableWidget::processRow(NetworkResourceTreeTO const& treeTO, std::vector<Column> const& columns)
{
    if (treeTO->isLeaf()) {
        _data->lastSessionData.registrate(treeTO->getLeaf().rawTO);
    }

    ImGui::TableNextRow(0, scale(BrowserGui::RowHeight));
    ImGui::TableNextColumn();

    auto selected = _data->selectedTreeTO == treeTO;
    if (ImGui::Selectable(
            "",
            &selected,
            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
            ImVec2(0, scale(BrowserGui::RowHeight) - ImGui::GetStyle().FramePadding.y))) {
        _data->selectedTreeTO = selected ? treeTO : nullptr;
    }
    ImGui::SameLine();

    pushTextColor(treeTO);

    columns.front().processField(treeTO);
    for (auto const& column : columns | std::views::drop(1)) {
        ImGui::TableNextColumn();
        column.processField(treeTO);
    }

    popTextColor();
}

void _BrowserTableWidget::processResourceNameField(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();

        processFolderTreeSymbols(treeTO, collapsedFolderNames);
        BrowserGui::DownloadButton(_data, leaf);
        ImGui::SameLine();
        if (_data->currentWorkspace.workspaceType == WorkspaceType_Private && leaf.rawTO->workspaceType != WorkspaceType_Private) {
            AlienGui::Text(ICON_FA_SHARE_ALT);
            AlienGui::Tooltip(leaf.rawTO->workspaceType == WorkspaceType_AlienProject ? "Visible in Featured" : "Visible in Community");
        }
        ImGui::SameLine();

        if (!_data->isOwner(treeTO) && _data->lastSessionData.isNew(leaf.rawTO)) {
            auto font = StyleRepository::get().getSmallBoldFont();
            auto origSize = font->Scale;
            font->Scale *= 0.65f;
            ImGui::PushFont(font);
            ImGui::PushStyleColor(ImGuiCol_Text, Const::BrowserResourceNewTextColor.Value);
            AlienGui::Text("NEW");
            ImGui::PopStyleColor();
            font->Scale = origSize;
            ImGui::PopFont();

            ImGui::SameLine();
        }

        BrowserGui::ShortenedText(leaf.leafName, true);
    } else {
        auto& folder = treeTO->getFolder();

        processFolderTreeSymbols(treeTO, collapsedFolderNames);
        BrowserGui::ShortenedText(treeTO->folderNames.back());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        std::string resourceTypeString = [&] {
            if (treeTO->type == NetworkResourceType_Simulation) {
                return folder.numLeafs == 1 ? "sim" : "sims";
            } else {
                return folder.numLeafs == 1 ? "genome" : "genomes";
            }
        }();
        AlienGui::Text("(" + std::to_string(folder.numLeafs) + " " + resourceTypeString + ")");
        ImGui::PopStyleColor();
    }
}

void _BrowserTableWidget::processFolderTreeSymbols(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames)
{
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserResourceSymbolColor);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0, 0, 0));
    auto const& treeSymbols = treeTO->treeSymbols;
    for (auto const& folderLine : treeSymbols) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGuiStyle& style = ImGui::GetStyle();
        switch (folderLine) {
        case FolderTreeSymbols::Expanded: {
            if (AlienGui::Button(ICON_FA_MINUS_SQUARE, 20.0f)) {
                collapsedFolderNames.insert(treeTO->folderNames);
                _recreateTreeTOs = true;
            }
        } break;
        case FolderTreeSymbols::Collapsed: {
            if (AlienGui::Button(ICON_FA_PLUS_SQUARE, 20.0f)) {
                collapsedFolderNames.erase(treeTO->folderNames);
                _recreateTreeTOs = true;
            }
        } break;
        case FolderTreeSymbols::Continue: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(BrowserGui::RowHeight) + style.FramePadding.y),
                Const::BrowserResourceLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::Branch: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(BrowserGui::RowHeight) + style.FramePadding.y),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f - 0.5f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y - scale(0.5f)),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f + 2.0f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y + scale(2.0f)),
                Const::BrowserResourceLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::End: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f - 0.5f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y - scale(0.5f)),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f + 2.0f), pos.y + scale(BrowserGui::RowHeight) / 2 - style.FramePadding.y + scale(2.0f)),
                Const::BrowserResourceLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::None: {
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        default: {
        } break;
        }
        ImGui::SameLine();
    }
    ImGui::PopStyleColor(2);
}

void _BrowserTableWidget::processDescriptionField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        BrowserGui::ShortenedText(leaf.rawTO->description);
    }
}

void _BrowserTableWidget::processReactionList(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();

        auto isAddReaction = AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS));
        AlienGui::Tooltip("Add a reaction", false);
        if (isAddReaction) {
            _data->activateEmojiPopup = true;
            _data->emojiPopupTO = treeTO;
        }

        // Calc remap which allows to show most frequent like type first
        std::map<int, int> remap;
        std::set<int> processedEmojiTypes;

        int index = 0;
        while (processedEmojiTypes.size() < leaf.rawTO->numLikesByEmojiType.size()) {
            int maxLikes = 0;
            std::optional<int> maxEmojiType;
            for (auto const& [emojiType, numLikes] : leaf.rawTO->numLikesByEmojiType) {
                if (!processedEmojiTypes.contains(emojiType) && numLikes > maxLikes) {
                    maxLikes = numLikes;
                    maxEmojiType = emojiType;
                }
            }
            processedEmojiTypes.insert(*maxEmojiType);
            remap.emplace(index, *maxEmojiType);
            ++index;
        }

        // Show like types with count
        int counter = 0;
        std::optional<int> toggleEmojiType;
        for (auto const& emojiType : remap | std::views::values) {
            auto numLikes = leaf.rawTO->numLikesByEmojiType.at(emojiType);

            ImGui::SameLine();
            AlienGui::Text(std::to_string(numLikes));
            if (emojiType < _data->emojis.size()) {
                ImGui::SameLine();
                auto const& emoji = _data->emojis.at(emojiType);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(7.0f));
                ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
                auto cursorPos = ImGui::GetCursorScreenPos();
                auto emojiWidth = scale(toFloat(emoji.width) / 2.5f);
                auto emojiHeight = scale(toFloat(emoji.height) / 2.5f);
                ImGui::PushID(emojiType);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(4.0f));
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - scale(3.0f));
                if (ImGui::ImageButton("reaction_emoji", (ImTextureID)(intptr_t)emoji.textureId, ImVec2(emojiWidth, emojiHeight), ImVec2(0, 0), ImVec2(1, 1))) {
                    toggleEmojiType = emojiType;
                }
                bool isLiked = _data->ownEmojiTypeBySimId.contains(leaf.rawTO->id) && _data->ownEmojiTypeBySimId.at(leaf.rawTO->id) == emojiType;
                if (isLiked) {
                    ImGui::GetWindowDrawList()->AddRect(
                        ImVec2(cursorPos.x, cursorPos.y),
                        ImVec2(cursorPos.x + emojiWidth, cursorPos.y + emojiHeight),
                        (ImU32)ImColor::HSV(0, 0, 1, 0.5f),
                        1.0f);
                }
                ImGui::PopStyleColor(2);
                ImGui::PopID();
                AlienGui::Tooltip([=, this] { return _data->getUserNamesToEmojiType(leaf.rawTO->id, emojiType); }, false);
            }

            // Separator except for last element
            if (++counter < leaf.rawTO->numLikesByEmojiType.size()) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(4.0f));
            }
        }
        if (toggleEmojiType) {
            _data->onToggleLike(treeTO, *toggleEmojiType);
        }
    } else {
        auto& folder = treeTO->getFolder();

        auto pos = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos({pos.x + scale(3.0f), pos.y});
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text("(" + std::to_string(folder.numReactions) + ")");
        ImGui::PopStyleColor();
    }
}

void _BrowserTableWidget::processTimestampField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienGui::Text(leaf.rawTO->timestamp);
    }
}

void _BrowserTableWidget::processUserNameField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        BrowserGui::ShortenedText(leaf.rawTO->userName);
    }
}

void _BrowserTableWidget::processNumDownloadsField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienGui::Text(AlienGui::TextParameters().text(std::to_string(leaf.rawTO->numDownloads)).rightAligned(true));
    }
}

void _BrowserTableWidget::processWidthField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienGui::Text(AlienGui::TextParameters().text(std::to_string(leaf.rawTO->width)).rightAligned(true));
    }
}

void _BrowserTableWidget::processHeightField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienGui::Text(AlienGui::TextParameters().text(std::to_string(leaf.rawTO->height)).rightAligned(true));
    }
}

void _BrowserTableWidget::processNumObjectsField(NetworkResourceTreeTO const& treeTO, bool kobjects)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        if (kobjects) {
            AlienGui::Text(StringHelper::format(leaf.rawTO->particles / 1000) + " K");
        } else {
            AlienGui::Text(AlienGui::TextParameters().text(StringHelper::format(leaf.rawTO->particles)).rightAligned(true));
        }
    }
}

void _BrowserTableWidget::processSizeField(NetworkResourceTreeTO const& treeTO, bool kbyte)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        if (kbyte) {
            AlienGui::Text(StringHelper::format(leaf.rawTO->contentSize / 1024) + " KB");
        } else {
            AlienGui::Text(StringHelper::format(leaf.rawTO->contentSize) + " Bytes");
        }
    }
}

void _BrowserTableWidget::processVersionField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienGui::Text(leaf.rawTO->version);
    }
}
