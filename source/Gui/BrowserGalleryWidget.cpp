#include "BrowserGalleryWidget.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>

#include <boost/range/adaptor/indexed.hpp>

#include <glad/glad.h>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/StringHelper.h>

#include <Network/NetworkResourceService.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/TaskProcessor.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserGui.h"
#include "OpenGLHelper.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr TilesPerPage = 100;
    auto constexpr MaxPicturesPerRequest = 32;
    auto constexpr BaseTileWidth = 190.0f;
    auto constexpr MinCardSizePercent = 60;
    auto constexpr MaxCardSizePercent = 200;
    auto constexpr TileSpacing = 8.0f;
    auto constexpr NumTileTextLines = 3;  // Path, name and the user with the date
    auto constexpr PictureAspectRatio = 2.0f / 3.0f;

    auto constexpr PlaceholderBarHeight = 5.0f;
    auto const PlaceholderBarWidthFactors = std::array{0.55f, 0.80f, 0.65f};

    auto constexpr SortingSwitcherWidth = 230.0f;
    auto constexpr CardSizeSliderWidth = 230.0f;
    auto constexpr MinCardSizeSliderWidth = 130.0f;

    auto constexpr TooltipDelay = 0.5f;  // In seconds
    auto constexpr TooltipValueWidth = 170.0f;
    auto constexpr TooltipWrapChars = 35.0f;
}

BrowserGalleryWidget _BrowserGalleryWidget::create(BrowserData const& data)
{
    return BrowserGalleryWidget(new _BrowserGalleryWidget(data));
}

_BrowserGalleryWidget::_BrowserGalleryWidget(BrowserData const& data)
    : _data(data)
{
    _pictureProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _sorting = GlobalSettings::get().getValue("windows.browser.gallery sorting", _sorting);
    _cardSizePercent = GlobalSettings::get().getValue("windows.browser.gallery card size", _cardSizePercent);
}

void _BrowserGalleryWidget::shutdown()
{
    GlobalSettings::get().setValue("windows.browser.gallery sorting", _sorting);
    GlobalSettings::get().setValue("windows.browser.gallery card size", _cardSizePercent);
}

void _BrowserGalleryWidget::processSorting()
{
    if (AlienGui::Switcher(
            AlienGui::SwitcherParameters()
                .name("Sort by")
                .width(SortingSwitcherWidth)
                .textWidth(55.0f)
                .values({std::string("Most reactions"), std::string("Newest"), std::string("Most downloads")}),
            &_sorting)) {
        _page = 0;
    }

    ImGui::SameLine();
    AlienGui::VerticalSeparator();
    ImGui::SameLine();

    auto cardSizeSliderWidth = std::clamp(scaleInverse(ImGui::GetContentRegionAvail().x), MinCardSizeSliderWidth, CardSizeSliderWidth);
    AlienGui::SliderInt(
        AlienGui::SliderIntParameters()
            .name("Card size")
            .width(cardSizeSliderWidth)
            .textWidth(75.0f)
            .min(MinCardSizePercent)
            .max(MaxCardSizePercent)
            .format("%d %%")
            .tooltip("Scale the preview cards. Larger cards mean fewer cards per row."),
        &_cardSizePercent);
}

void _BrowserGalleryWidget::process()
{
    ImGui::PushID("Gallery");

    auto entries = getSortedEntries();
    _numEntries = toInt(entries.size());

    _numPages = std::max(1, (_numEntries + TilesPerPage - 1) / TilesPerPage);
    _page = std::clamp(_page, 0, _numPages - 1);

    auto firstIndex = _page * TilesPerPage;
    auto lastIndex = std::min(_numEntries, firstIndex + TilesPerPage);
    auto pageEntries = std::vector<NetworkResourceRawTO>(entries.begin() + firstIndex, entries.begin() + lastIndex);

    requestMissingPictures(pageEntries);

    // The tile height follows the tile width, so an appearing scrollbar must not change the available width
    if (ImGui::BeginChild(
            "##tiles", {0, ImGui::GetContentRegionAvail().y - scale(BrowserGui::WorkspaceBottomSpace)}, false, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto layout = calcTileLayout();

        for (auto const& [index, rawTO] : pageEntries | boost::adaptors::indexed(0)) {
            if (index % layout.numColumns != 0) {
                ImGui::SameLine(0, layout.horizontalSpacing);
            }
            ImGui::PushID(toInt(index));
            processTile(rawTO, layout.tileWidth);
            ImGui::PopID();
        }
    }
    ImGui::EndChild();

    ImGui::PopID();
}

void _BrowserGalleryWidget::processPlaceholderTiles()
{
    ImGui::PushID("GalleryPlaceholder");

    if (ImGui::BeginChild(
            "##tiles", {0, ImGui::GetContentRegionAvail().y - scale(BrowserGui::WorkspaceBottomSpace)}, false, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto layout = calcTileLayout();
        auto tileHeight = calcTileHeight(layout.tileWidth);
        auto numRows = std::max(1, toInt(std::ceil(ImGui::GetContentRegionAvail().y / (tileHeight + ImGui::GetStyle().ItemSpacing.y))));

        for (auto const& index : std::views::iota(0, layout.numColumns * numRows)) {
            if (index % layout.numColumns != 0) {
                ImGui::SameLine(0, layout.horizontalSpacing);
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

void _BrowserGalleryWidget::processPendingRequests()
{
    _pictureProcessor->process();
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

void _BrowserGalleryWidget::invalidatePicture(std::string const& resourceId)
{
    auto findResult = _pictureBySimId.find(resourceId);
    if (findResult == _pictureBySimId.end()) {
        return;
    }
    if (findResult->second.has_value()) {
        glDeleteTextures(1, &findResult->second->textureId);
    }
    _pictureBySimId.erase(findResult);
}

namespace
{
    // Drawn inside the tile: child windows are rendered after their parent and would cover a border drawn around them
    void processSelectionBorder(ImVec2 const& tileSize)
    {
        auto thickness = scale(2.0f);
        auto tileMin = ImGui::GetWindowPos();
        auto tileMax = ImVec2{tileMin.x + tileSize.x, tileMin.y + tileSize.y};

        auto drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(tileMin, tileMax, false);
        drawList->AddRect(
            {tileMin.x + thickness / 2, tileMin.y + thickness / 2},
            {tileMax.x - thickness / 2, tileMax.y - thickness / 2},
            (ImU32)Const::AccentColor,
            ImGui::GetStyle().ChildRounding,
            0,
            thickness);
        drawList->PopClipRect();
    }

    NetworkResourceTreeTO createLeafTreeTO(NetworkResourceRawTO const& rawTO)
    {
        auto result = std::make_shared<_NetworkResourceTreeTO>();
        result->type = rawTO->resourceType;
        result->folderNames = NetworkResourceService::get().getFolderNames(rawTO->resourceName);
        result->node = BrowserLeaf{.leafName = NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName), .rawTO = rawTO};
        return result;
    }
}

_BrowserGalleryWidget::TileLayout _BrowserGalleryWidget::calcTileLayout() const
{
    auto horizontalSpacing = scale(TileSpacing);
    auto availableWidth = ImGui::GetContentRegionAvail().x;
    auto tileWidth = std::floor(std::min(availableWidth, scale(BaseTileWidth) * toFloat(_cardSizePercent) / 100.0f));
    return {
        .tileWidth = tileWidth,
        .horizontalSpacing = horizontalSpacing,
        .numColumns = std::max(1, toInt((availableWidth + horizontalSpacing) / (tileWidth + horizontalSpacing)))};
}

float _BrowserGalleryWidget::calcTileHeight(float tileWidth) const
{
    auto const& style = ImGui::GetStyle();
    auto pictureHeight = (tileWidth - style.WindowPadding.x * 2) * PictureAspectRatio;
    return std::floor(
        pictureHeight + NumTileTextLines * ImGui::GetTextLineHeight() + ImGui::GetFrameHeight()  // Button row
        + (NumTileTextLines + 1) * style.ItemSpacing.y + style.WindowPadding.y * 2);
}

void _BrowserGalleryWidget::processTile(NetworkResourceRawTO const& rawTO, float tileWidth)
{
    _data->lastSessionData.registrate(rawTO);

    auto tileHeight = calcTileHeight(tileWidth);

    auto buttonHovered = false;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)Const::PanelColor);
    if (ImGui::BeginChild("##tile", {tileWidth, tileHeight}, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        auto textWidth = ImGui::GetContentRegionAvail().x;
        processPicture(rawTO, textWidth);

        auto folderNames = NetworkResourceService::get().getFolderNames(rawTO->resourceName);
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserLeafTextColor);
        AlienGui::Text(AlienGui::TextParameters().text(NetworkResourceService::get().concatenateFolderName(folderNames, true)).truncate(true));
        ImGui::PopStyleColor();

        if (_data->currentWorkspace.workspaceType == WorkspaceType_Private && rawTO->workspaceType != WorkspaceType_Private) {
            AlienGui::Text(ICON_FA_SHARE_ALT);
            AlienGui::Tooltip(rawTO->workspaceType == WorkspaceType_AlienProject ? "Visible in Featured" : "Visible in Community");
            ImGui::SameLine();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserResourceTextColor);
        AlienGui::Text(AlienGui::TextParameters()
                           .text(NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName))
                           .style(AlienGui::TextStyle::Bold)
                           .truncate(true));
        ImGui::PopStyleColor();

        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text(rawTO->userName);
        ImGui::SameLine();
        AlienGui::Text(AlienGui::TextParameters().text(rawTO->timestamp.substr(0, 10)).rightAligned(true));
        ImGui::PopStyleColor();

        BrowserGui::DownloadButton(_data, BrowserLeaf{.leafName = rawTO->resourceName, .rawTO = rawTO});
        ImGui::SameLine();
        processReactionButton(rawTO);
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text(AlienGui::TextParameters().text(ICON_FA_DOWNLOAD " " + std::to_string(rawTO->numDownloads)).rightAligned(true));
        ImGui::PopStyleColor();

        buttonHovered = ImGui::IsAnyItemHovered();

        if (_data->isSelected(rawTO)) {
            processSelectionBorder({tileWidth, tileHeight});
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    auto tileMin = ImGui::GetItemRectMin();
    auto tileMax = ImGui::GetItemRectMax();
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && ImGui::IsMouseHoveringRect(tileMin, tileMax)) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            onSelectEntry(rawTO);
        }
        if (!buttonHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            _data->onDownloadResource(BrowserLeaf{.leafName = rawTO->resourceName, .rawTO = rawTO});
        }

        // The buttons of the tile have tooltips of their own
        if (!buttonHovered) {
            if (_hoveredTileId != rawTO->id) {
                _hoveredTileId = rawTO->id;
                _hoveredTileTime = 0;
            } else {
                _hoveredTileTime += ImGui::GetIO().DeltaTime;
            }
            if (_hoveredTileTime > TooltipDelay) {
                processTileTooltip(rawTO);
            }
        }
    }
}

namespace
{
    void processTooltipLabel(std::string const& label)
    {
        ImGui::SameLine(scale(TooltipValueWidth));
        AlienGui::Text(AlienGui::TextParameters().text(label).style(AlienGui::TextStyle::Decent));
    }

    void processTooltipRow(std::string const& label, std::string const& value)
    {
        AlienGui::Text(value);
        processTooltipLabel(label);
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
        auto pictureHeight = contentWidth * PictureAspectRatio;
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

void _BrowserGalleryWidget::processTileTooltip(NetworkResourceRawTO const& rawTO)
{
    ImGui::BeginTooltip();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextTooltipColor.Value);

    auto findResult = _pictureBySimId.find(rawTO->id);
    if (findResult != _pictureBySimId.end() && findResult->second.has_value()) {
        auto const& picture = *findResult->second;
        ImGui::Image((ImTextureID)(intptr_t)picture.textureId, {scale(toFloat(picture.width)), scale(toFloat(picture.height))});
    }

    auto folderNames = NetworkResourceService::get().getFolderNames(rawTO->resourceName);
    if (!folderNames.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserLeafTextColor);
        AlienGui::Text(NetworkResourceService::get().concatenateFolderName(folderNames, true));
        ImGui::PopStyleColor();
    }
    AlienGui::Text(AlienGui::TextParameters().text(NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName)).style(AlienGui::TextStyle::Bold));

    if (!rawTO->description.empty()) {
        AlienGui::Separator();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * TooltipWrapChars);
        ImGui::TextUnformatted(rawTO->description.c_str());
        ImGui::PopTextWrapPos();
    }

    AlienGui::Separator();

    auto isSimulation = rawTO->resourceType == NetworkResourceType_Simulation;
    processTooltipRow("User", rawTO->userName);
    processTooltipRow("Timestamp", rawTO->timestamp);

    if (rawTO->numLikesByEmojiType.empty()) {
        AlienGui::Text("-");
    } else {
        for (auto const& [emojiType, numLikes] : rawTO->numLikesByEmojiType) {
            if (emojiType < toInt(_data->emojis.size())) {
                auto const& emoji = _data->emojis.at(emojiType);
                ImGui::Image((ImTextureID)(intptr_t)emoji.textureId, {scale(toFloat(emoji.width) / 2.5f), scale(toFloat(emoji.height) / 2.5f)});
                ImGui::SameLine();
            }
            AlienGui::Text(std::to_string(numLikes));
            ImGui::SameLine();
        }
    }
    processTooltipLabel("Reactions");

    processTooltipRow("Downloads", std::to_string(rawTO->numDownloads));
    if (isSimulation) {
        processTooltipRow("World size", std::to_string(rawTO->width) + " x " + std::to_string(rawTO->height));
        processTooltipRow("Objects", StringHelper::format(rawTO->particles / 1000) + " K");
        processTooltipRow("File size", StringHelper::format(rawTO->contentSize / 1024) + " KB");
    } else {
        processTooltipRow("Cells", StringHelper::format(rawTO->particles));
        processTooltipRow("File size", StringHelper::format(rawTO->contentSize) + " Bytes");
    }
    processTooltipRow("Version", rawTO->version);

    ImGui::PopStyleColor();
    ImGui::EndTooltip();
}

void _BrowserGalleryWidget::processPicture(NetworkResourceRawTO const& rawTO, float width)
{
    auto height = width * PictureAspectRatio;
    auto pos = ImGui::GetCursorScreenPos();

    auto findResult = _pictureBySimId.find(rawTO->id);
    if (findResult != _pictureBySimId.end() && findResult->second.has_value()) {
        ImGui::Image((ImTextureID)(intptr_t)findResult->second->textureId, {width, height});
        return;
    }

    ImGui::GetWindowDrawList()->AddRectFilled(pos, {pos.x + width, pos.y + height}, (ImU32)Const::BackgroundColor);
    auto text = hasPreviewPictures() && findResult == _pictureBySimId.end() ? std::string("loading...") : std::string("no preview");
    auto textSize = ImGui::CalcTextSize(text.c_str());
    ImGui::GetWindowDrawList()->AddText({pos.x + (width - textSize.x) / 2, pos.y + (height - textSize.y) / 2}, (ImU32)Const::TextDecentColor, text.c_str());
    ImGui::Dummy({width, height});
}

void _BrowserGalleryWidget::processReactionButton(NetworkResourceRawTO const& rawTO)
{
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserAddReactionButtonTextColor);
    auto isAddReaction = BrowserGui::ActionButton(ICON_FA_HEART " " + std::to_string(rawTO->getTotalLikes()));
    ImGui::PopStyleColor();

    if (ImGui::IsItemHovered()) {
        processReactionTooltip(rawTO);
    }
    if (isAddReaction) {
        _data->activateEmojiPopup = true;
        _data->emojiPopupTO = createLeafTreeTO(rawTO);
    }
}

void _BrowserGalleryWidget::processReactionTooltip(NetworkResourceRawTO const& rawTO)
{
    ImGui::BeginTooltip();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextTooltipColor.Value);
    if (rawTO->numLikesByEmojiType.empty()) {
        AlienGui::Text("Add a reaction");
    } else {
        for (auto const& [emojiType, numLikes] : rawTO->numLikesByEmojiType) {
            if (emojiType < toInt(_data->emojis.size())) {
                auto const& emoji = _data->emojis.at(emojiType);
                ImGui::Image((ImTextureID)(intptr_t)emoji.textureId, {scale(toFloat(emoji.width) / 2.5f), scale(toFloat(emoji.height) / 2.5f)});
                ImGui::SameLine();
            }
            AlienGui::Text(std::to_string(numLikes) + "   " + _data->getUserNamesToEmojiType(rawTO->id, emojiType));
        }
    }
    ImGui::PopStyleColor();
    ImGui::EndTooltip();
}

std::string _BrowserGalleryWidget::getPageText() const
{
    auto firstEntry = _numEntries > 0 ? _page * TilesPerPage + 1 : 0;
    auto lastEntry = std::min(_numEntries, (_page + 1) * TilesPerPage);
    return std::to_string(firstEntry) + " - " + std::to_string(lastEntry) + " of " + std::to_string(_numEntries);
}

std::vector<NetworkResourceRawTO> _BrowserGalleryWidget::getSortedEntries() const
{
    auto const& workspace = _data->workspaces.at(_data->currentWorkspace);

    std::vector<NetworkResourceRawTO> result;
    for (auto const& rawTO : workspace.rawTOs) {
        if (_data->filter.empty() || rawTO->matchWithFilter(_data->filter)) {
            result.emplace_back(rawTO);
        }
    }

    std::ranges::sort(result, [this](NetworkResourceRawTO const& left, NetworkResourceRawTO const& right) {
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

void _BrowserGalleryWidget::requestMissingPictures(std::vector<NetworkResourceRawTO> const& pageEntries)
{
    if (!hasPreviewPictures() || _pictureProcessor->pendingTasks()) {
        return;
    }

    std::vector<std::string> simIds;
    for (auto const& rawTO : pageEntries) {
        if (!_pictureBySimId.contains(rawTO->id) && toInt(simIds.size()) < MaxPicturesPerRequest) {
            simIds.emplace_back(rawTO->id);
        }
    }
    if (simIds.empty()) {
        return;
    }

    _pictureProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleGetSimulationPictures(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, GetSimulationPicturesRequestData{.simIds = simIds});
        },
        [&, simIds](auto const& requestId) {
            auto data = _PersisterFacade::get()->fetchGetSimulationPicturesData(requestId);
            for (auto const& simId : simIds) {
                auto findResult = data.jpgBySimId.find(simId);
                std::optional<TextureData> picture;
                if (findResult != data.jpgBySimId.end() && !findResult->second.empty()) {
                    try {
                        picture = OpenGLHelper::loadTextureFromMemory(findResult->second);
                    } catch (std::exception const&) {
                        log(Priority::Important, "browser: preview picture of simulation " + simId + " could not be decoded");
                    }
                }
                _pictureBySimId.insert_or_assign(simId, picture);
            }
        },
        [&, simIds](auto const&) {
            for (auto const& simId : simIds) {
                _pictureBySimId.insert_or_assign(simId, std::nullopt);
            }
        });
}

bool _BrowserGalleryWidget::hasPreviewPictures() const
{
    return _data->currentWorkspace.resourceType == NetworkResourceType_Simulation;
}

void _BrowserGalleryWidget::onSelectEntry(NetworkResourceRawTO const& rawTO)
{
    _data->selectedTreeTO = createLeafTreeTO(rawTO);
}
