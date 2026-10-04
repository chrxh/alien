#include "MarkdownRenderer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>
#include <cfloat>

#include <stb_image.h>

#include <Base/MarkdownParser.h>

#include "AlienGui.h"
#include "StyleService.h"
#include "TextureService.h"

namespace
{
    constexpr float LineHeightFactor = 1.35f;
    constexpr float ParagraphSpacing = 8.0f;
    constexpr float ListIndent = 22.0f;
    constexpr float ListItemSpacing = 2.0f;
    constexpr float BoxPadding = 8.0f;
    constexpr float NoteBarWidth = 3.0f;
    constexpr float BoxRounding = 3.0f;
    constexpr float ImageOversampling = 2.0f;
}

namespace
{
    float getSpacingAfter(MarkdownBlock const& block)
    {
        if (std::holds_alternative<MarkdownHeading>(block) || std::holds_alternative<MarkdownListItem>(block)) {
            return ListItemSpacing;
        }
        if (std::holds_alternative<MarkdownRule>(block)) {
            return 0.0f;
        }
        return ParagraphSpacing;
    }
}

std::optional<std::string> MarkdownRenderer::render(MarkdownDocument const& document, std::filesystem::path const& basePath)
{
    if (&document != _renderedDocument) {
        releaseTextures();
        _renderedDocument = &document;
    }
    _basePath = basePath;
    _clickedLink.reset();
    _hoveredLink = std::exchange(_nextHoveredLink, std::nullopt);
    _textureCreatedInFrame = false;

    ImGui::PushID(&document);
    for (auto const& [index, block] : std::views::enumerate(document.blocks)) {
        ImGui::PushID(static_cast<int>(index));
        if (auto heading = std::get_if<MarkdownHeading>(&block)) {
            renderHeading(*heading);
        } else if (auto paragraph = std::get_if<MarkdownParagraph>(&block)) {
            renderParagraph(*paragraph);
        } else if (auto listItem = std::get_if<MarkdownListItem>(&block)) {
            renderListItem(*listItem);
        } else if (auto image = std::get_if<MarkdownImage>(&block)) {
            renderImage(*image);
        } else if (auto codeBlock = std::get_if<MarkdownCodeBlock>(&block)) {
            renderCodeBlock(*codeBlock);
        } else if (auto note = std::get_if<MarkdownNote>(&block)) {
            renderNote(*note);
        } else if (auto table = std::get_if<MarkdownTable>(&block)) {
            renderTable(*table);
        } else {
            AlienGui::Separator();
        }
        auto spacing = getSpacingAfter(block);
        if (spacing > 0.0f) {
            ImGui::Dummy({0.0f, scale(spacing)});
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    _pendingAnchor.reset();
    return _clickedLink;
}

void MarkdownRenderer::scrollToAnchor(std::string const& anchor)
{
    _pendingAnchor = anchor;
}

void MarkdownRenderer::releaseTextures()
{
    for (auto& imageInfo : _imageInfoByPath | std::views::values) {
        if (imageInfo.has_value() && imageInfo->texture.has_value()) {
            TextureService::get().deleteTexture(*imageInfo->texture);
            imageInfo->texture.reset();
        }
    }
}

void MarkdownRenderer::renderHeading(MarkdownHeading const& heading)
{
    auto& styleService = StyleService::get();
    if (heading.level > 1) {
        ImGui::Dummy({0.0f, scale(heading.level == 2 ? 10.0f : 4.0f)});
    }
    if (_pendingAnchor == heading.anchor) {
        ImGui::SetScrollHereY(0.0f);
        _pendingAnchor.reset();
    }
    auto font = heading.level == 1 ? styleService.getMediumBoldFont() : styleService.getSmallBoldFont();
    auto color = heading.level <= 2 ? Const::HeadlineColor : Const::TextStrongColor;
    renderTextFlow(heading.spans, {.font = font, .color = color});
    if (heading.level == 1) {
        AlienGui::Separator();
    }
}

void MarkdownRenderer::renderParagraph(MarkdownParagraph const& paragraph, float rightPadding)
{
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + scale(ListIndent) * static_cast<float>(paragraph.listIndentLevel));
    renderTextFlow(paragraph.spans, {.font = StyleService::get().getDefaultFont(), .color = ImGui::GetColorU32(ImGuiCol_Text), .rightPadding = rightPadding});
}

void MarkdownRenderer::renderListItem(MarkdownListItem const& listItem, float rightPadding)
{
    auto font = StyleService::get().getDefaultFont();
    auto color = ImGui::GetColorU32(ImGuiCol_Text);
    auto indent = scale(ListIndent) * static_cast<float>(listItem.depth + 1);

    ImGui::Indent(indent);
    auto pos = ImGui::GetCursorScreenPos();
    auto drawList = ImGui::GetWindowDrawList();
    if (listItem.number.has_value()) {
        auto marker = std::to_string(*listItem.number) + ".";
        auto markerWidth = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, marker.c_str()).x;
        drawList->AddText(font, font->FontSize, {pos.x - markerWidth - scale(6.0f), pos.y}, color, marker.c_str());
    } else {
        drawList->AddCircleFilled({pos.x - scale(11.0f), pos.y + font->Ascent * 0.65f}, scale(2.5f), Const::HeadlineColor);
    }
    renderTextFlow(listItem.spans, {.font = font, .color = color, .rightPadding = rightPadding});
    ImGui::Unindent(indent);
}

namespace
{
    std::optional<TextureData> loadTexture(std::filesystem::path const& path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            return std::nullopt;
        }
        try {
            std::string encodedImage{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
            return TextureService::get().loadTextureFromMemory(encodedImage);
        } catch (std::exception const&) {
            return std::nullopt;
        }
    }
}

void MarkdownRenderer::renderImage(MarkdownImage const& image)
{
    auto path = _basePath / image.source;
    auto& imageInfo = getImageInfo(path);
    if (!imageInfo.has_value()) {
        renderTextFlow({{.text = "Image not found: " + image.source}}, {.font = StyleService::get().getDefaultFont(), .color = Const::WarningColor});
        return;
    }
    auto availableWidth = ImGui::GetContentRegionAvail().x;
    auto width = std::min(availableWidth, scale(static_cast<float>(imageInfo->width) / ImageOversampling));
    auto height = width * static_cast<float>(imageInfo->height) / static_cast<float>(imageInfo->width);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availableWidth - width) / 2);

    // Decoding is deferred until the image becomes visible and limited to one image per frame
    if (!imageInfo->texture.has_value() && !_textureCreatedInFrame && ImGui::IsRectVisible({width, height})) {
        _textureCreatedInFrame = true;
        imageInfo->texture = loadTexture(path);
        if (!imageInfo->texture.has_value()) {
            imageInfo.reset();
        }
    }
    if (imageInfo.has_value() && imageInfo->texture.has_value()) {
        ImGui::Image(imageInfo->texture->textureId, {width, height});
    } else {
        ImGui::Dummy({width, height});
    }

    if (!image.caption.empty()) {
        auto font = StyleService::get().getDefaultFont();
        auto captionWidth = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, image.caption.c_str()).x;
        if (captionWidth < availableWidth) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availableWidth - captionWidth) / 2);
        }
        renderTextFlow({{.text = image.caption}}, {.font = font, .color = Const::TextDimColor});
    }
}

void MarkdownRenderer::renderCodeBlock(MarkdownCodeBlock const& codeBlock, float rightPadding)
{
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + scale(ListIndent) * static_cast<float>(codeBlock.listIndentLevel));

    auto font = StyleService::get().getMonospaceMediumFont();
    auto padding = scale(BoxPadding);
    auto textSize = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, codeBlock.text.c_str());
    auto min = ImGui::GetCursorScreenPos();
    auto max = ImVec2{min.x + ImGui::GetContentRegionAvail().x - rightPadding, min.y + textSize.y + padding * 2};

    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(min, max, Const::InputColor, scale(BoxRounding));
    drawList->PushClipRect(min, max, true);
    drawList->AddText(font, font->FontSize, {min.x + padding, min.y + padding}, Const::TextStrongColor, codeBlock.text.c_str());
    drawList->PopClipRect();

    ImGui::Dummy({0.0f, max.y - min.y});
}

void MarkdownRenderer::renderNote(MarkdownNote const& note)
{
    auto padding = scale(BoxPadding);
    auto barWidth = scale(NoteBarWidth);
    auto start = ImGui::GetCursorScreenPos();
    auto width = ImGui::GetContentRegionAvail().x;

    auto drawList = ImGui::GetWindowDrawList();
    drawList->ChannelsSplit(2);
    drawList->ChannelsSetCurrent(1);

    ImGui::SetCursorScreenPos({start.x, start.y + padding});
    ImGui::Indent(padding + barWidth);
    for (auto const& [index, block] : std::views::enumerate(note.blocks)) {
        if (index > 0) {
            ImGui::Dummy({0.0f, scale(ListItemSpacing)});
        }
        if (auto paragraph = std::get_if<MarkdownParagraph>(&block)) {
            renderParagraph(*paragraph, padding);
        } else if (auto listItem = std::get_if<MarkdownListItem>(&block)) {
            renderListItem(*listItem, padding);
        } else if (auto codeBlock = std::get_if<MarkdownCodeBlock>(&block)) {
            renderCodeBlock(*codeBlock, padding);
        }
    }
    ImGui::Unindent(padding + barWidth);
    auto endY = ImGui::GetCursorScreenPos().y - ImGui::GetStyle().ItemSpacing.y + padding;

    drawList->ChannelsSetCurrent(0);
    drawList->AddRectFilled(start, {start.x + width, endY}, Const::RaisedColor, scale(BoxRounding));
    drawList->AddRectFilled(start, {start.x + barWidth, endY}, Const::AccentColor, scale(BoxRounding), ImDrawFlags_RoundCornersLeft);
    drawList->ChannelsMerge();

    ImGui::SetCursorScreenPos({start.x, endY});
}

namespace
{
    ImFont* getSpanFont(MarkdownSpan const& span, ImFont* baseFont)
    {
        auto& styleService = StyleService::get();
        if (span.code) {
            return styleService.getMonospaceMediumFont();
        }
        if (span.bold && baseFont == styleService.getDefaultFont()) {
            return styleService.getSmallBoldFont();
        }
        return baseFont;
    }

    float calcSingleLineWidth(MarkdownSpans const& spans, ImFont* baseFont)
    {
        auto result = 0.0f;
        for (auto const& span : spans) {
            auto font = getSpanFont(span, baseFont);
            result += font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, span.text.c_str()).x;
        }
        return result;
    }
}

void MarkdownRenderer::renderTable(MarkdownTable const& table)
{
    if (table.header.empty()) {
        return;
    }
    auto flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##table", static_cast<int>(table.header.size()), flags)) {
        for (auto const& headerCell : table.header) {
            ImGui::TableSetupColumn(MarkdownParser::toPlainText(headerCell).c_str());
        }
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        auto font = StyleService::get().getDefaultFont();
        for (auto const& row : table.rows) {
            ImGui::TableNextRow();
            for (auto const& [cell, alignment] : std::views::zip(row, table.alignments)) {
                ImGui::TableNextColumn();
                if (alignment == MarkdownAlignment::Right) {
                    auto cellWidth = calcSingleLineWidth(cell, font);
                    auto availableWidth = ImGui::GetContentRegionAvail().x;
                    if (cellWidth < availableWidth) {
                        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availableWidth - cellWidth);
                    }
                }
                renderTextFlow(cell, {.font = font, .color = ImGui::GetColorU32(ImGuiCol_Text)});
            }
        }
        ImGui::EndTable();
    }
}

namespace
{
    ImU32 getSpanColor(MarkdownSpan const& span, ImU32 baseColor, bool hoveredLink)
    {
        if (span.link.has_value()) {
            return hoveredLink ? Const::TextStrongColor : Const::AccentColor;
        }
        if (span.bold || span.italic || span.code) {
            return Const::TextStrongColor;
        }
        return baseColor;
    }

    struct TextPiece
    {
        MarkdownSpan const* span = nullptr;
        ImFont* font = nullptr;
        std::string_view text;
        float width = 0.0f;
        float spaceWidthBefore = 0.0f;
        int numLineBreaksBefore = 0;
    };

    // Pieces contain no whitespace. Pieces without whitespace between them form a word, even if their spans differ.
    std::vector<TextPiece> splitIntoPieces(MarkdownSpans const& spans, ImFont* baseFont)
    {
        std::vector<TextPiece> result;
        auto spaceWidthBefore = 0.0f;
        auto numLineBreaksBefore = 0;
        for (auto const& span : spans) {
            auto font = getSpanFont(span, baseFont);
            auto remainingText = std::string_view(span.text);
            while (!remainingText.empty()) {
                if (remainingText.front() == '\n') {
                    ++numLineBreaksBefore;
                    remainingText.remove_prefix(1);
                    continue;
                }
                if (remainingText.front() == ' ') {
                    spaceWidthBefore = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, " ").x;
                    remainingText.remove_prefix(1);
                    continue;
                }
                auto text = remainingText.substr(0, remainingText.find_first_of(" \n"));
                auto width = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, text.data(), text.data() + text.size()).x;
                result.emplace_back(TextPiece{
                    .span = &span,
                    .font = font,
                    .text = text,
                    .width = width,
                    .spaceWidthBefore = spaceWidthBefore,
                    .numLineBreaksBefore = numLineBreaksBefore});
                spaceWidthBefore = 0.0f;
                numLineBreaksBefore = 0;
                remainingText.remove_prefix(text.size());
            }
        }
        return result;
    }

    bool continuesWord(TextPiece const&, TextPiece const& nextPiece)
    {
        return nextPiece.spaceWidthBefore == 0.0f && nextPiece.numLineBreaksBefore == 0;
    }

    struct PlacedPiece
    {
        MarkdownSpan const* span = nullptr;
        float endX = 0.0f;
        float y = 0.0f;
    };
}

void MarkdownRenderer::renderTextFlow(MarkdownSpans const& spans, TextFlowStyle const& style)
{
    auto drawList = ImGui::GetWindowDrawList();
    auto origin = ImGui::GetCursorScreenPos();
    auto maxX = origin.x + ImGui::GetContentRegionAvail().x - style.rightPadding;
    auto lineHeight = std::round(style.font->FontSize * LineHeightFactor);
    auto windowHovered = ImGui::IsWindowHovered();
    auto pos = origin;
    auto maxLineX = origin.x;
    std::optional<PlacedPiece> previousPiece;

    auto pieces = splitIntoPieces(spans, style.font);
    for (auto const& word : pieces | std::views::chunk_by(continuesWord)) {
        auto const& firstPiece = word.front();
        if (firstPiece.numLineBreaksBefore > 0) {
            pos = {origin.x, pos.y + lineHeight * static_cast<float>(firstPiece.numLineBreaksBefore)};
        } else if (pos.x > origin.x) {
            pos.x += firstPiece.spaceWidthBefore;
        }
        auto wordWidth = 0.0f;
        for (auto const& piece : word) {
            wordWidth += piece.width;
        }
        if (pos.x + wordWidth > maxX + 1.0f && pos.x > origin.x) {
            pos = {origin.x, pos.y + lineHeight};
        }

        for (auto const& piece : word) {
            auto const& span = *piece.span;
            auto pieceMin = ImVec2{pos.x, pos.y};
            auto pieceMax = ImVec2{pos.x + piece.width, pos.y + lineHeight};

            // Code backgrounds and link underlines continue across the spaces between pieces on the same line
            auto continuesLine = previousPiece.has_value() && previousPiece->y == pos.y;
            if (span.code) {
                auto continuesBackground = continuesLine && previousPiece->span == &span;
                drawList->AddRectFilled(
                    {continuesBackground ? previousPiece->endX : pieceMin.x - scale(2.0f), pieceMin.y},
                    {pieceMax.x + scale(2.0f), pieceMax.y - scale(2.0f)},
                    Const::InputColor,
                    scale(2.0f),
                    continuesBackground ? ImDrawFlags_RoundCornersRight : ImDrawFlags_None);
            }
            auto color = getSpanColor(span, style.color, span.link.has_value() && span.link == _hoveredLink);
            drawList->AddText(
                piece.font,
                piece.font->FontSize,
                {pos.x, pos.y + style.font->Ascent - piece.font->Ascent},
                color,
                piece.text.data(),
                piece.text.data() + piece.text.size());

            if (span.link.has_value()) {
                auto underlineY = pos.y + style.font->Ascent + scale(2.0f);
                auto underlineStartX = continuesLine && previousPiece->span->link == span.link ? previousPiece->endX : pieceMin.x;
                drawList->AddLine({underlineStartX, underlineY}, {pieceMax.x, underlineY}, color);
                if (windowHovered && ImGui::IsMouseHoveringRect(pieceMin, pieceMax)) {
                    _nextHoveredLink = span.link;
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    if (span.link->starts_with("http")) {
                        ImGui::SetTooltip("%s", span.link->c_str());
                    }
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        _clickedLink = span.link;
                    }
                }
            }
            pos.x += piece.width;
            maxLineX = std::max(maxLineX, pos.x);
            previousPiece = PlacedPiece{.span = &span, .endX = pos.x, .y = pos.y};
        }
    }
    ImGui::Dummy({maxLineX - origin.x, pos.y + lineHeight - origin.y});
}

std::optional<MarkdownRenderer::ImageInfo>& MarkdownRenderer::getImageInfo(std::filesystem::path const& path)
{
    auto [iterator, inserted] = _imageInfoByPath.try_emplace(path.lexically_normal());
    if (inserted) {
        auto width = 0;
        auto height = 0;
        auto numChannels = 0;
        if (stbi_info(path.string().c_str(), &width, &height, &numChannels) != 0 && width > 0 && height > 0) {
            iterator->second = ImageInfo{.width = width, .height = height};
        }
    }
    return iterator->second;
}
