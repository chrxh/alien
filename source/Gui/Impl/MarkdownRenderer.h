#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include <imgui.h>

#include <Base/Interface/MarkdownDocument.h>

#include <Rendering/Interface/TextureData.h>

#include "Definitions.h"

class MarkdownRenderer
{
public:
    // Image sources are resolved relative to basePath. Returns the target of a clicked link.
    std::optional<std::string> render(MarkdownDocument const& document, std::filesystem::path const& basePath);

    void scrollToAnchor(std::string const& anchor);

    void releaseTextures();

private:
    void renderHeading(MarkdownHeading const& heading);
    void renderParagraph(MarkdownParagraph const& paragraph, float rightPadding = 0.0f);
    void renderListItem(MarkdownListItem const& listItem, float rightPadding = 0.0f);
    void renderImage(MarkdownImage const& image);
    void renderCodeBlock(MarkdownCodeBlock const& codeBlock, float rightPadding = 0.0f);
    void renderNote(MarkdownNote const& note);
    void renderTable(MarkdownTable const& table);

    struct TextFlowStyle
    {
        ImFont* font = nullptr;
        ImU32 color = 0;
        float rightPadding = 0.0f;
    };
    void renderTextFlow(MarkdownSpans const& spans, TextFlowStyle const& style);

    struct ImageInfo
    {
        int width = 0;
        int height = 0;
        std::optional<TextureData> texture;
    };
    std::optional<ImageInfo>& getImageInfo(std::filesystem::path const& path);

    std::filesystem::path _basePath;
    std::optional<std::string> _pendingAnchor;
    std::optional<std::string> _clickedLink;
    std::optional<std::string> _hoveredLink;
    std::optional<std::string> _nextHoveredLink;

    MarkdownDocument const* _renderedDocument = nullptr;
    bool _textureCreatedInFrame = false;
    std::map<std::filesystem::path, std::optional<ImageInfo>> _imageInfoByPath;
};
