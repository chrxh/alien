#include "BrowserGalleryWidget.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <Network/NetworkResourceService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserHelper.h"
#include "BrowserPictureCache.h"
#include "StyleService.h"

namespace
{
    auto constexpr TilesPerPage = 100;
    auto constexpr PreferredTileWidth = 190.0f;
    auto constexpr TileSpacing = 8.0f;
    auto constexpr NumTileTextLines = 3;  // Path, name and a line of facts

    auto constexpr BadgeMargin = 6.0f;

    auto constexpr PlaceholderBarHeight = 5.0f;
    auto const PlaceholderBarWidthFactors = std::array{0.55f, 0.80f, 0.65f};

    auto constexpr SortingSwitcherWidth = 230.0f;
    auto constexpr SortingTextWidth = 55.0f;

    auto constexpr TimestampLengthWithoutSeconds = 16;  // e.g. 2026-11-13 14:02
    auto constexpr DateLength = 10;
}

BrowserGalleryWidget _BrowserGalleryWidget::create(BrowserData const& data, BrowserPictureCache const& pictureCache)
{
    return BrowserGalleryWidget(new _BrowserGalleryWidget(data, pictureCache));
}

_BrowserGalleryWidget::_BrowserGalleryWidget(BrowserData const& data, BrowserPictureCache const& pictureCache)
    : _data(data)
    , _pictureCache(pictureCache)
{
    auto& settings = GlobalSettings::get();
    _sorting = settings.getValue("windows.browser.gallery sorting", _sorting);
    _localSorting = settings.getValue("windows.browser.gallery local sorting", _localSorting);
}

void _BrowserGalleryWidget::shutdown()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("windows.browser.gallery sorting", _sorting);
    settings.setValue("windows.browser.gallery local sorting", _localSorting);
}

void _BrowserGalleryWidget::processSorting()
{
    auto isLocal = isLocalWorkspace(_data->getCurrentWorkspaceId().workspaceType);
    auto values = isLocal ? std::vector{std::string("Newest"), std::string("Name")}
                          : std::vector{std::string("Most reactions"), std::string("Newest"), std::string("Most downloads")};
    if (AlienGui::Switcher(
            AlienGui::SwitcherParameters().name("Sort by").width(SortingSwitcherWidth).textWidth(SortingTextWidth).values(values),
            isLocal ? &_localSorting : &_sorting)) {
        _page = 0;
    }
}

void _BrowserGalleryWidget::process()
{
    ImGui::PushID("Gallery");

    auto entries = getSortedEntries();
    _numEntries = toInt(entries.size());

    _numPages = std::max(1, (_numEntries + TilesPerPage - 1) / TilesPerPage);
    if (_data->isScrollToSelection()) {
        auto findResult = std::ranges::find_if(entries, [&](NetworkResourceRawTO const& rawTO) { return _data->isSelected(rawTO); });
        if (findResult != entries.end()) {
            _page = toInt(std::distance(entries.begin(), findResult)) / TilesPerPage;
        }
    }
    _page = std::clamp(_page, 0, _numPages - 1);

    auto firstIndex = _page * TilesPerPage;
    auto lastIndex = std::min(_numEntries, firstIndex + TilesPerPage);
    auto pageEntries = std::vector<NetworkResourceRawTO>(entries.begin() + firstIndex, entries.begin() + lastIndex);

    _pictureCache->request(pageEntries);

    // The tile height follows the tile width, so an appearing scrollbar must not change the available width
    if (ImGui::BeginChild(
            "##tiles", {0, ImGui::GetContentRegionAvail().y - BrowserHelper::calcFooterHeight()}, false, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        processKeyboardNavigation(pageEntries);

        auto layout = calcTileLayout();
        _numColumns = layout.numColumns;
        if (pageEntries.empty()) {
            processEmptyHint();
        }
        auto rowPosX = ImGui::GetCursorPosX();
        for (auto const& [index, rawTO] : pageEntries | boost::adaptors::indexed(0)) {
            if (auto column = toInt(index) % layout.numColumns; column != 0) {
                ImGui::SameLine(rowPosX + layout.columnOffsets.at(column));
            }
            ImGui::PushID(toInt(index));
            processTile(rawTO, layout.tileWidth);
            ImGui::PopID();
        }

        // A click beside the tiles removes the selection, which shows the workspace instead of a resource
        auto scrollbarPosX = ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize;
        if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::GetMousePos().x < scrollbarPosX) {
            _data->setSelectedTreeTO(nullptr);
        }
    }
    ImGui::EndChild();

    ImGui::PopID();
}

void _BrowserGalleryWidget::processPlaceholderTiles()
{
    ImGui::PushID("GalleryPlaceholder");

    if (ImGui::BeginChild(
            "##tiles", {0, ImGui::GetContentRegionAvail().y - BrowserHelper::calcFooterHeight()}, false, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto layout = calcTileLayout();
        auto tileHeight = calcTileHeight(layout.tileWidth);
        auto numRows = std::max(1, toInt(std::ceil(ImGui::GetContentRegionAvail().y / (tileHeight + ImGui::GetStyle().ItemSpacing.y))));

        auto rowPosX = ImGui::GetCursorPosX();
        for (auto const& index : std::views::iota(0, layout.numColumns * numRows)) {
            if (auto column = index % layout.numColumns; column != 0) {
                ImGui::SameLine(rowPosX + layout.columnOffsets.at(column));
            }
            ImGui::PushID(index);
            processPlaceholderTile(layout.tileWidth);
            ImGui::PopID();
        }
    }
    ImGui::EndChild();

    ImGui::PopID();
}

void _BrowserGalleryWidget::processPaging()
{
    ImGui::BeginDisabled(_page == 0);
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_ANGLE_DOUBLE_LEFT).tooltip("First page"))) {
        _page = 0;
    }
    ImGui::SameLine();
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_ANGLE_LEFT).tooltip("Previous page"))) {
        --_page;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienGui::Text(getPageText());

    ImGui::SameLine();
    ImGui::BeginDisabled(_page >= _numPages - 1);
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_ANGLE_RIGHT).tooltip("Next page"))) {
        ++_page;
    }
    ImGui::SameLine();
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_ANGLE_DOUBLE_RIGHT).tooltip("Last page"))) {
        _page = _numPages - 1;
    }
    ImGui::EndDisabled();
}

float _BrowserGalleryWidget::getSortingWidth() const
{
    return scale(SortingSwitcherWidth);
}

float _BrowserGalleryWidget::getPagerWidth() const
{
    auto const& style = ImGui::GetStyle();
    auto getButtonWidth = [&](std::string const& icon) { return ImGui::CalcTextSize(icon.c_str()).x + style.FramePadding.x * 2; };
    return getButtonWidth(ICON_FA_ANGLE_DOUBLE_LEFT) + getButtonWidth(ICON_FA_ANGLE_LEFT) + getButtonWidth(ICON_FA_ANGLE_RIGHT)
        + getButtonWidth(ICON_FA_ANGLE_DOUBLE_RIGHT) + ImGui::CalcTextSize(getPageText().c_str()).x + style.ItemSpacing.x * 4;
}

void _BrowserGalleryWidget::resetPage()
{
    _page = 0;
}

// The tiles fill their row. The number of columns is the one that brings their width closest to the preferred one.
_BrowserGalleryWidget::TileLayout _BrowserGalleryWidget::calcTileLayout() const
{
    auto minSpacing = scale(TileSpacing);
    auto preferredTileWidth = scale(PreferredTileWidth);
    auto availableWidth = ImGui::GetContentRegionAvail().x;
    auto calcTileWidth = [&](int numColumns) {
        return std::max(1.0f, std::floor((availableWidth - minSpacing * toFloat(numColumns - 1)) / toFloat(numColumns)));
    };

    auto numColumns = std::max(1, toInt((availableWidth + minSpacing) / (preferredTileWidth + minSpacing)));
    if (std::abs(calcTileWidth(numColumns + 1) - preferredTileWidth) < std::abs(calcTileWidth(numColumns) - preferredTileWidth)) {
        ++numColumns;
    }
    auto tileWidth = calcTileWidth(numColumns);

    // The pixels that the whole tile widths leave over widen the gaps, so that the last column ends at the right border
    auto columnDistance = numColumns > 1 ? (availableWidth - tileWidth) / toFloat(numColumns - 1) : 0.0f;
    std::vector<float> columnOffsets;
    for (auto const& column : std::views::iota(0, numColumns)) {
        columnOffsets.emplace_back(std::round(toFloat(column) * columnDistance));
    }
    return {.tileWidth = tileWidth, .columnOffsets = columnOffsets, .numColumns = numColumns};
}

float _BrowserGalleryWidget::calcTileHeight(float tileWidth) const
{
    auto const& style = ImGui::GetStyle();
    auto pictureHeight = BrowserHelper::calcPictureHeight(tileWidth - style.WindowPadding.x * 2);
    return std::floor(pictureHeight + NumTileTextLines * (ImGui::GetTextLineHeight() + style.ItemSpacing.y) + style.WindowPadding.y * 2);
}

void _BrowserGalleryWidget::processTile(NetworkResourceRawTO const& rawTO, float tileWidth)
{
    auto isLocal = isLocalWorkspace(rawTO->workspaceType);
    if (!isLocal) {
        _data->registrateAsSeen(rawTO);
    }

    auto tileHeight = calcTileHeight(tileWidth);

    auto heartHovered = false;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)Const::PanelColor);
    if (ImGui::BeginChild("##tile", {tileWidth, tileHeight}, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        auto contentWidth = ImGui::GetContentRegionAvail().x;
        auto picturePos = ImGui::GetCursorScreenPos();
        BrowserHelper::Picture(_pictureCache->find(rawTO), contentWidth);
        processBadges(rawTO, picturePos);

        // A simulation of this computer shows the reactions to its upload
        if (auto reactedRawTO = isLocal ? _data->findUpload(rawTO) : rawTO) {
            heartHovered =
                BrowserHelper::PictureHeart(_data, reactedRawTO, {picturePos.x + contentWidth, picturePos.y + BrowserHelper::calcPictureHeight(contentWidth)});
        }

        auto folderNames = NetworkResourceService::get().getFolderNames(rawTO->resourceName);
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserLeafTextColor);
        AlienGui::Text(AlienGui::TextParameters().text(NetworkResourceService::get().concatenateFolderName(folderNames, true)).truncate(true));
        ImGui::PopStyleColor();

        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserResourceTextColor);
        AlienGui::Text(AlienGui::TextParameters()
                           .text(NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName))
                           .style(AlienGui::TextStyle::Bold)
                           .truncate(true));
        ImGui::PopStyleColor();

        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        if (rawTO->workspaceType == WorkspaceType_Autosave) {
            AlienGui::Text(AlienGui::TextParameters().text(rawTO->timestamp.substr(0, TimestampLengthWithoutSeconds)).truncate(true));
        } else {
            AlienGui::Text(isLocal ? StringHelper::formatFileSize(rawTO->contentSize) : rawTO->userName);
            ImGui::SameLine();
            AlienGui::Text(AlienGui::TextParameters().text(rawTO->timestamp.substr(0, DateLength)).rightAligned(true));
        }
        ImGui::PopStyleColor();

        if (_data->isSelected(rawTO)) {
            BrowserHelper::TileSelectionBorder({tileWidth, tileHeight});
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    auto tileMin = ImGui::GetItemRectMin();
    auto tileMax = ImGui::GetItemRectMax();
    if (_data->isScrollToSelection() && _data->isSelected(rawTO)) {
        auto viewMin = ImGui::GetWindowPos().y;
        auto viewMax = viewMin + ImGui::GetWindowHeight();
        if (tileMin.y < viewMin || tileMax.y > viewMax) {
            ImGui::SetScrollHereY(0.5f);
        }
    }

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && ImGui::IsMouseHoveringRect(tileMin, tileMax)) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            onSelectEntry(rawTO);
        }
        if (!heartHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            _data->onOpenResource(BrowserLeaf{.leafName = rawTO->resourceName, .rawTO = rawTO});
        }
    }
}

void _BrowserGalleryWidget::processPlaceholderTile(float tileWidth)
{
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)Const::PanelColor);
    if (ImGui::BeginChild(
            "##placeholderTile", {tileWidth, calcTileHeight(tileWidth)}, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        auto contentWidth = ImGui::GetContentRegionAvail().x;
        auto drawList = ImGui::GetWindowDrawList();

        auto picturePos = ImGui::GetCursorScreenPos();
        auto pictureHeight = BrowserHelper::calcPictureHeight(contentWidth);
        drawList->AddRectFilled(picturePos, {picturePos.x + contentWidth, picturePos.y + pictureHeight}, Const::BrowserPlaceholderTilePictureColor);
        ImGui::Dummy({contentWidth, pictureHeight});

        auto barHeight = scale(PlaceholderBarHeight);
        for (auto const& widthFactor : PlaceholderBarWidthFactors) {
            auto barPos = ImGui::GetCursorScreenPos();
            auto barOffsetY = (ImGui::GetTextLineHeight() - barHeight) / 2;
            drawList->AddRectFilled(
                {barPos.x, barPos.y + barOffsetY},
                {barPos.x + contentWidth * widthFactor, barPos.y + barOffsetY + barHeight},
                Const::BrowserPlaceholderTileBarColor);
            ImGui::Dummy({contentWidth, ImGui::GetTextLineHeight()});
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void _BrowserGalleryWidget::processBadges(NetworkResourceRawTO const& rawTO, ImVec2 const& picturePos)
{
    if (_data->isNew(rawTO)) {
        BrowserHelper::PictureBadge(
            {picturePos.x + scale(BadgeMargin), picturePos.y + scale(BadgeMargin)},
            "NEW",
            Const::BrowserResourceNewTextColor,
            Const::BrowserNewBadgeBorderColor);
    }
}

void _BrowserGalleryWidget::processEmptyHint()
{
    auto text = [&]() -> std::string {
        if (!_data->getFilter().empty()) {
            return "Nothing matches the filter";
        }
        switch (_data->getCurrentWorkspaceId().workspaceType) {
        case WorkspaceType_Local:
            return _data->getCurrentWorkspaceId().resourceType == NetworkResourceType_Simulation
                ? "No simulations in this folder yet. Save (Ctrl+S) stores the current simulation here."
                : "No genomes in this folder yet. Save genome in the tool bar stores the genome of the genome editor here.";
        case WorkspaceType_Autosave:
            return "No autosaves yet";
        case WorkspaceType_Liked:
            return "Nothing liked yet. Click the heart of an entry to collect it here.";
        default:
            return "Nothing here yet";
        }
    }();

    auto pos = ImGui::GetCursorScreenPos();
    auto size = ImGui::GetContentRegionAvail();
    auto textSize = ImGui::CalcTextSize(text.c_str());
    ImGui::GetWindowDrawList()->AddText(
        {pos.x + std::max(0.0f, (size.x - textSize.x) / 2), pos.y + std::max(0.0f, (size.y - textSize.y) / 2)}, (ImU32)Const::TextDecentColor, text.c_str());
}

void _BrowserGalleryWidget::processKeyboardNavigation(std::vector<NetworkResourceRawTO> const& pageEntries)
{
    // Only the tiles react to the keys, since Enter opens a resource and replaces the running simulation
    if (pageEntries.empty() || !ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) || ImGui::GetIO().WantTextInput) {
        return;
    }
    auto offset = [&]() -> int {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
            return -1;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
            return 1;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            return -_numColumns;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            return _numColumns;
        }
        return 0;
    }();

    auto findResult = std::ranges::find_if(pageEntries, [&](NetworkResourceRawTO const& rawTO) { return _data->isSelected(rawTO); });
    if (findResult == pageEntries.end()) {
        if (offset != 0) {
            onSelectEntry(pageEntries.front());
            _data->setScrollToSelection(true);
        }
        return;
    }
    if (offset != 0) {
        auto newIndex = toInt(std::distance(pageEntries.begin(), findResult)) + offset;
        if (newIndex >= 0 && newIndex < toInt(pageEntries.size())) {
            onSelectEntry(pageEntries.at(newIndex));
            _data->setScrollToSelection(true);
        }
    } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) {
        _data->onOpenResource(BrowserLeaf{.leafName = (*findResult)->resourceName, .rawTO = *findResult});
    }
}

std::string _BrowserGalleryWidget::getPageText() const
{
    auto firstEntry = _numEntries > 0 ? _page * TilesPerPage + 1 : 0;
    auto lastEntry = std::min(_numEntries, (_page + 1) * TilesPerPage);
    return std::to_string(firstEntry) + " - " + std::to_string(lastEntry) + " of " + std::to_string(_numEntries);
}

std::vector<NetworkResourceRawTO> _BrowserGalleryWidget::getSortedEntries() const
{
    auto const& workspace = _data->getCurrentWorkspace();

    std::vector<NetworkResourceRawTO> result;
    for (auto const& rawTO : workspace.rawTOs) {
        if (_data->matchesFilter(rawTO)) {
            result.emplace_back(rawTO);
        }
    }

    auto isLocal = isLocalWorkspace(_data->getCurrentWorkspaceId().workspaceType);
    std::ranges::stable_sort(result, [&](NetworkResourceRawTO const& left, NetworkResourceRawTO const& right) {
        if (isLocal) {
            if (_localSorting == LocalGallerySorting_Name) {
                return StringHelper::naturalLess(left->resourceName, right->resourceName);
            }
            return left->timestamp > right->timestamp;
        }
        if (_sorting == GallerySorting_Newest) {
            return left->timestamp > right->timestamp;
        }
        if (_sorting == GallerySorting_MostDownloads) {
            return left->numDownloads > right->numDownloads;
        }
        return left->getTotalLikes() > right->getTotalLikes();
    });
    return result;
}

void _BrowserGalleryWidget::onSelectEntry(NetworkResourceRawTO const& rawTO)
{
    _data->setSelectedTreeTO(BrowserHelper::createLeafTreeTO(rawTO));
}
