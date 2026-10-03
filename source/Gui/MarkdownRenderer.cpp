#include "MarkdownRenderer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <ranges>
#include <string_view>
#include <utility>
#include <cfloat>

#include <glad/glad.h>

#include "AlienGui.h"
#include "MarkdownParser.h"
#include "OpenGLHelper.h"
#include "StyleService.h"

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

std::optional<std::string> MarkdownRenderer::render(MarkdownDocument const& document, std::filesystem::path const& basePath)
{
    _basePath = basePath;
    _clickedLink.reset();
    _hoveredLink = std::exchange(_nextHoveredLink, std::nullopt);

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
        ImGui::PopID();
    }
    _pendingAnchor.reset();
    return _clickedLink;
}

void MarkdownRenderer::scrollToAnchor(std::string const& anchor)
{
    _pendingAnchor = anchor;
}

void MarkdownRenderer::releaseTextures()
{
    for (auto const& texture : _textureByPath | std::views::values) {
        if (texture.has_value()) {
            glDeleteTextures(1, &texture->textureId);
        }
    }
    _textureByPath.clear();
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
    ImGui::Dummy({0.0f, scale(ListItemSpacing)});
}

void MarkdownRenderer::renderParagraph(MarkdownParagraph const& paragraph)
{
    renderTextFlow(paragraph.spans, {.font = StyleService::get().getDefaultFont(), .color = ImGui::GetColorU32(ImGuiCol_Text)});
    ImGui::Dummy({0.0f, scale(ParagraphSpacing)});
}

void MarkdownRenderer::renderListItem(MarkdownListItem const& listItem)
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
    renderTextFlow(listItem.spans, {.font = font, .color = color});
    ImGui::Unindent(indent);
    ImGui::Dummy({0.0f, scale(ListItemSpacing)});
}

void MarkdownRenderer::renderImage(MarkdownImage const& image)
{
    auto texture = getTexture(_basePath / image.source);
    if (!texture.has_value()) {
        renderTextFlow({{.text = "Image not found: " + image.source}}, {.font = StyleService::get().getDefaultFont(), .color = Const::WarningColor});
        ImGui::Dummy({0.0f, scale(ParagraphSpacing)});
        return;
    }
    auto availableWidth = ImGui::GetContentRegionAvail().x;
    auto width = std::min(availableWidth, scale(static_cast<float>(texture->width) / ImageOversampling));
    auto height = width * static_cast<float>(texture->height) / static_cast<float>(texture->width);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availableWidth - width) / 2);
    ImGui::Image((ImTextureID)(intptr_t)texture->textureId, {width, height});

    if (!image.caption.empty()) {
        auto font = StyleService::get().getDefaultFont();
        auto captionWidth = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, image.caption.c_str()).x;
        if (captionWidth < availableWidth) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availableWidth - captionWidth) / 2);
        }
        renderTextFlow({{.text = image.caption}}, {.font = font, .color = Const::TextDimColor});
    }
    ImGui::Dummy({0.0f, scale(ParagraphSpacing)});
}

void MarkdownRenderer::renderCodeBlock(MarkdownCodeBlock const& codeBlock)
{
    auto font = StyleService::get().getMonospaceMediumFont();
    auto padding = scale(BoxPadding);
    auto textSize = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, codeBlock.text.c_str());
    auto min = ImGui::GetCursorScreenPos();
    auto max = ImVec2{min.x + ImGui::GetContentRegionAvail().x, min.y + textSize.y + padding * 2};

    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(min, max, Const::InputColor, scale(BoxRounding));
    drawList->PushClipRect(min, max, true);
    drawList->AddText(font, font->FontSize, {min.x + padding, min.y + padding}, Const::TextStrongColor, codeBlock.text.c_str());
    drawList->PopClipRect();

    ImGui::Dummy({0.0f, max.y - min.y});
    ImGui::Dummy({0.0f, scale(ParagraphSpacing)});
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
    for (auto const& paragraph : note.paragraphs) {
        renderTextFlow(paragraph, {.font = StyleService::get().getDefaultFont(), .color = ImGui::GetColorU32(ImGuiCol_Text), .rightPadding = padding});
    }
    ImGui::Unindent(padding + barWidth);
    auto endY = ImGui::GetCursorScreenPos().y - ImGui::GetStyle().ItemSpacing.y + padding;

    drawList->ChannelsSetCurrent(0);
    drawList->AddRectFilled(start, {start.x + width, endY}, Const::RaisedColor, scale(BoxRounding));
    drawList->AddRectFilled(start, {start.x + barWidth, endY}, Const::AccentColor, scale(BoxRounding), ImDrawFlags_RoundCornersLeft);
    drawList->ChannelsMerge();

    ImGui::SetCursorScreenPos({start.x, endY});
    ImGui::Dummy({0.0f, scale(ParagraphSpacing)});
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
    ImGui::Dummy({0.0f, scale(ParagraphSpacing)});
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

    for (auto const& span : spans) {
        auto font = getSpanFont(span, style.font);
        auto fontSize = font->FontSize;
        auto yOffset = style.font->Ascent - font->Ascent;
        auto spaceWidth = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, " ").x;
        auto color = getSpanColor(span, style.color, span.link.has_value() && span.link == _hoveredLink);

        std::optional<ImVec2> linkSpaceStart;
        auto remainingText = std::string_view(span.text);
        while (!remainingText.empty()) {
            if (remainingText.front() == '\n') {
                pos = {origin.x, pos.y + lineHeight};
                linkSpaceStart.reset();
                remainingText.remove_prefix(1);
                continue;
            }
            if (remainingText.front() == ' ') {
                if (pos.x > origin.x) {
                    if (span.link.has_value() && !linkSpaceStart.has_value()) {
                        linkSpaceStart = pos;
                    }
                    pos.x += spaceWidth;
                }
                remainingText.remove_prefix(1);
                continue;
            }
            auto word = remainingText.substr(0, remainingText.find_first_of(" \n"));
            auto wordWidth = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, word.data(), word.data() + word.size()).x;
            if (pos.x + wordWidth > maxX + 1.0f && pos.x > origin.x) {
                pos = {origin.x, pos.y + lineHeight};
            }

            auto wordMin = ImVec2{pos.x, pos.y};
            auto wordMax = ImVec2{pos.x + wordWidth, pos.y + lineHeight};
            if (span.code) {
                drawList->AddRectFilled(
                    {wordMin.x - scale(2.0f), wordMin.y}, {wordMax.x + scale(2.0f), wordMax.y - scale(2.0f)}, Const::InputColor, scale(2.0f));
            }
            drawList->AddText(font, fontSize, {pos.x, pos.y + yOffset}, color, word.data(), word.data() + word.size());

            if (span.link.has_value()) {
                auto underlineY = pos.y + style.font->Ascent + scale(2.0f);
                auto underlineStartX = linkSpaceStart.has_value() && linkSpaceStart->y == pos.y ? linkSpaceStart->x : wordMin.x;
                drawList->AddLine({underlineStartX, underlineY}, {wordMax.x, underlineY}, color);
                if (windowHovered && ImGui::IsMouseHoveringRect(wordMin, wordMax)) {
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
            pos.x += wordWidth;
            maxLineX = std::max(maxLineX, pos.x);
            linkSpaceStart.reset();
            remainingText.remove_prefix(word.size());
        }
    }
    ImGui::Dummy({maxLineX - origin.x, pos.y + lineHeight - origin.y});
}

std::optional<TextureData> MarkdownRenderer::getTexture(std::filesystem::path const& path)
{
    auto [iterator, inserted] = _textureByPath.try_emplace(path.lexically_normal());
    if (inserted) {
        std::ifstream stream(path, std::ios::binary);
        if (stream) {
            try {
                std::string encodedImage{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
                iterator->second = OpenGLHelper::loadTextureFromMemory(encodedImage);
            } catch (std::exception const&) {
            }
        }
    }
    return iterator->second;
}
