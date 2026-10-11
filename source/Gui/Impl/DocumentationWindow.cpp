#include "DocumentationWindow.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <ranges>
#include <cctype>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Interface/GlobalSettings.h>
#include <Base/Interface/MarkdownParser.h>
#include <Base/Interface/Resources.h>
#include <Base/Interface/WebLinkHelper.h>

#include "AlienGui.h"
#include "StyleService.h"
#include "WindowController.h"

namespace
{
    constexpr float NavigationWidth = 220.0f;
    constexpr float MinNavigationWidth = 80.0f;
    constexpr float MinContentWidth = 200.0f;
    constexpr float BottomSpace = 50.0f;
    constexpr float PartSpacing = 10.0f;
    constexpr float SectionIndent = 12.0f;
    constexpr size_t SnippetLeadLength = 30;
    constexpr size_t SnippetLength = 100;
    constexpr auto ContentsFilename = "contents.md";
}

DocumentationWindow::DocumentationWindow()
    : AlienWindow("Documentation", "windows.documentation", true, true, {420.0f, 120.0f}, {1040.0f, 720.0f})
{}

void DocumentationWindow::initIntern()
{
    _showAfterStartup = _on;
    _navigationWidth =
        GlobalSettings::get().getValue("windows.documentation.navigation width", scale(NavigationWidth)) * WindowController::get().getContentScaleCorrection();
    loadChapters();

    auto chapterFilename = GlobalSettings::get().getValue("windows.documentation.chapter", std::string());
    auto findResult = std::ranges::find(_chapters, chapterFilename, &Chapter::filename);
    if (findResult != _chapters.end()) {
        _currentChapterIndex = std::distance(_chapters.begin(), findResult);
    }
}

void DocumentationWindow::shutdownIntern()
{
    _on = _showAfterStartup;
    GlobalSettings::get().setValue("windows.documentation.navigation width", _navigationWidth);
    if (!_chapters.empty()) {
        GlobalSettings::get().setValue("windows.documentation.chapter", _chapters.at(_currentChapterIndex).filename);
    }
    _renderer.releaseTextures();
}

void DocumentationWindow::processIntern()
{
    if (ImGui::BeginChild("##documentation", {0, ImGui::GetContentRegionAvail().y - scale(BottomSpace)}, 0, ImGuiWindowFlags_NoScrollbar)) {
        // The stored width only changes by dragging, so that a temporarily narrow window does not shrink it permanently
        auto maxNavigationWidth = std::max(scale(MinNavigationWidth), ImGui::GetContentRegionAvail().x - scale(MinContentWidth));
        auto navigationWidth = std::clamp(_navigationWidth, scale(MinNavigationWidth), maxNavigationWidth);
        processNavigation(navigationWidth);

        ImGui::SameLine();
        auto draggedNavigationWidth = navigationWidth;
        AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters(), draggedNavigationWidth);
        if (draggedNavigationWidth != navigationWidth) {
            _navigationWidth = draggedNavigationWidth;
        }

        ImGui::SameLine();
        processContent();
    }
    ImGui::EndChild();

    AlienGui::Separator();
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Show after startup"), _showAfterStartup);
}

void DocumentationWindow::processBackground()
{
    if (!isShown()) {
        _renderer.releaseTextures();
    }
}

namespace
{
    std::string readFile(std::filesystem::path const& path)
    {
        std::ifstream stream(path);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    void appendText(std::string& target, std::string const& text)
    {
        if (text.empty()) {
            return;
        }
        if (!target.empty()) {
            target += ' ';
        }
        auto start = target.size();
        target += text;
        std::ranges::replace(target.begin() + start, target.end(), '\n', ' ');
    }

    std::string getPlainText(MarkdownBlock const& block)
    {
        if (auto heading = std::get_if<MarkdownHeading>(&block)) {
            return MarkdownParser::toPlainText(heading->spans);
        }
        if (auto paragraph = std::get_if<MarkdownParagraph>(&block)) {
            return MarkdownParser::toPlainText(paragraph->spans);
        }
        if (auto listItem = std::get_if<MarkdownListItem>(&block)) {
            return MarkdownParser::toPlainText(listItem->spans);
        }
        if (auto image = std::get_if<MarkdownImage>(&block)) {
            return image->caption;
        }
        if (auto codeBlock = std::get_if<MarkdownCodeBlock>(&block)) {
            return codeBlock->text;
        }
        std::string result;
        if (auto note = std::get_if<MarkdownNote>(&block)) {
            for (auto const& noteBlock : note->blocks) {
                appendText(result, std::visit([](auto const& value) { return getPlainText(MarkdownBlock(value)); }, noteBlock));
            }
        }
        if (auto table = std::get_if<MarkdownTable>(&block)) {
            for (auto const& cell : table->header) {
                appendText(result, MarkdownParser::toPlainText(cell));
            }
            for (auto const& cell : table->rows | std::views::join) {
                appendText(result, MarkdownParser::toPlainText(cell));
            }
        }
        return result;
    }
}

void DocumentationWindow::loadChapters()
{
    _chapters.clear();

    // The table of contents lists the chapters as links, grouped under part headings
    auto contents = MarkdownParser::parse(readFile(Const::DocsPath / ContentsFilename));
    std::string part;
    for (auto const& contentsBlock : contents.blocks) {
        if (auto heading = std::get_if<MarkdownHeading>(&contentsBlock)) {
            part = MarkdownParser::toPlainText(heading->spans);
            continue;
        }
        auto listItem = std::get_if<MarkdownListItem>(&contentsBlock);
        if (!listItem) {
            continue;
        }
        auto linkSpan = std::ranges::find_if(listItem->spans, [](auto const& span) { return span.link.has_value(); });
        if (linkSpan == listItem->spans.end()) {
            continue;
        }

        Chapter chapter{.filename = *linkSpan->link, .title = linkSpan->text, .part = part};
        chapter.document = MarkdownParser::parse(readFile(Const::DocsPath / chapter.filename));
        chapter.sections.emplace_back(Section{.title = chapter.title});
        for (auto const& block : chapter.document.blocks) {
            auto heading = std::get_if<MarkdownHeading>(&block);
            if (heading && heading->level == 1) {
                continue;
            }
            if (heading && heading->level == 2) {
                chapter.sections.emplace_back(Section{.title = MarkdownParser::toPlainText(heading->spans), .anchor = heading->anchor});
                continue;
            }
            appendText(chapter.sections.back().text, getPlainText(block));
        }
        _chapters.emplace_back(std::move(chapter));
    }
}

namespace
{
    std::string toLowerCase(std::string text)
    {
        std::ranges::transform(text, text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    std::vector<std::string> splitIntoTerms(std::string const& text)
    {
        std::vector<std::string> result;
        for (auto const& term : toLowerCase(text) | std::views::split(' ')) {
            if (!term.empty()) {
                result.emplace_back(term.begin(), term.end());
            }
        }
        return result;
    }

    // The snippet is cut at spaces so that multi-byte characters stay intact
    std::string createSnippet(std::string const& text, std::string const& lowerCaseText, std::vector<std::string> const& terms)
    {
        auto matchPos = std::string::npos;
        for (auto const& term : terms) {
            matchPos = std::min(matchPos, lowerCaseText.find(term));
        }
        if (matchPos == std::string::npos) {
            matchPos = 0;
        }

        size_t start = 0;
        if (matchPos > SnippetLeadLength) {
            auto spacePos = text.find(' ', matchPos - SnippetLeadLength);
            start = spacePos < matchPos ? spacePos + 1 : matchPos;
        }
        auto end = text.size();
        if (start + SnippetLength < text.size()) {
            auto spacePos = text.rfind(' ', start + SnippetLength);
            end = spacePos != std::string::npos && spacePos > matchPos ? spacePos : start + SnippetLength;
        }

        auto result = text.substr(start, end - start);
        if (start > 0) {
            result = "..." + result;
        }
        if (end < text.size()) {
            result += "...";
        }
        return result;
    }
}

void DocumentationWindow::updateSearchResults()
{
    _searchResults.clear();

    auto terms = splitIntoTerms(_searchText);
    if (terms.empty()) {
        return;
    }

    // Sections whose title contains all terms are listed first
    std::vector<SearchResult> textMatches;
    for (auto const& [chapterIndex, chapter] : std::views::enumerate(_chapters)) {
        for (auto const& [sectionIndex, section] : std::views::enumerate(chapter.sections)) {
            auto lowerCaseTitle = toLowerCase(section.title);
            auto lowerCaseText = toLowerCase(section.text);
            if (!std::ranges::all_of(terms, [&](auto const& term) { return lowerCaseTitle.contains(term) || lowerCaseText.contains(term); })) {
                continue;
            }
            SearchResult result{
                .chapterIndex = static_cast<size_t>(chapterIndex),
                .sectionIndex = static_cast<size_t>(sectionIndex),
                .snippet = createSnippet(section.text, lowerCaseText, terms)};
            if (std::ranges::all_of(terms, [&](auto const& term) { return lowerCaseTitle.contains(term); })) {
                _searchResults.emplace_back(std::move(result));
            } else {
                textMatches.emplace_back(std::move(result));
            }
        }
    }
    _searchResults.insert(_searchResults.end(), textMatches.begin(), textMatches.end());
}

void DocumentationWindow::processNavigation(float width)
{
    if (ImGui::BeginChild("##navigation", {width, 0})) {
        if (AlienGui::InputFilter(AlienGui::InputFilterParameters().hint("Search"), _searchText)) {
            updateSearchResults();
        }
        if (ImGui::BeginChild("##navigationList", {0, 0})) {
            if (_searchText.empty()) {
                processTableOfContents();
            } else {
                processSearchResults();
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
}

void DocumentationWindow::processTableOfContents()
{
    std::optional<std::string> currentPart;
    for (auto const& [chapterIndex, chapter] : std::views::enumerate(_chapters)) {
        ImGui::PushID(static_cast<int>(chapterIndex));

        if (chapter.part != currentPart) {
            if (currentPart.has_value()) {
                ImGui::Dummy({0.0f, scale(PartSpacing)});
            }
            currentPart = chapter.part;
            ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
            ImGui::PushFont(StyleService::get().getSmallBoldFont());
            ImGui::TextUnformatted(chapter.part.c_str());
            ImGui::PopFont();
            ImGui::PopStyleColor();
        }

        auto isCurrentChapter = static_cast<size_t>(chapterIndex) == _currentChapterIndex;
        ImGui::PushFont(isCurrentChapter ? StyleService::get().getSmallBoldFont() : StyleService::get().getDefaultFont());
        if (ImGui::Selectable(chapter.title.c_str(), isCurrentChapter)) {
            openChapter(chapterIndex);
        }
        ImGui::PopFont();

        if (isCurrentChapter) {
            ImGui::Indent(scale(SectionIndent));
            ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
            for (auto const& [sectionIndex, section] : chapter.sections | std::views::enumerate | std::views::drop(1)) {
                ImGui::PushID(static_cast<int>(sectionIndex));
                if (ImGui::Selectable(section.title.c_str())) {
                    openChapter(chapterIndex, section.anchor);
                }
                ImGui::PopID();
            }
            ImGui::PopStyleColor();
            ImGui::Unindent(scale(SectionIndent));
        }
        ImGui::PopID();
    }
}

void DocumentationWindow::processSearchResults()
{
    if (_searchResults.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        ImGui::TextUnformatted("No matches found.");
        ImGui::PopStyleColor();
        return;
    }
    for (auto const& [index, result] : std::views::enumerate(_searchResults)) {
        ImGui::PushID(static_cast<int>(index));
        auto const& chapter = _chapters.at(result.chapterIndex);
        auto const& section = chapter.sections.at(result.sectionIndex);

        ImGui::PushFont(StyleService::get().getSmallBoldFont());
        if (ImGui::Selectable(section.title.c_str())) {
            openChapter(result.chapterIndex, section.anchor.empty() ? std::nullopt : std::make_optional(section.anchor));
        }
        ImGui::PopFont();

        ImGui::PushTextWrapPos(0.0f);
        if (result.sectionIndex > 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, Const::HeadlineColor.Value);
            ImGui::TextUnformatted(chapter.title.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        ImGui::TextUnformatted(result.snippet.c_str());
        ImGui::PopStyleColor();
        ImGui::PopTextWrapPos();

        ImGui::Spacing();
        ImGui::PopID();
    }
}

void DocumentationWindow::processContent()
{
    if (ImGui::BeginChild("##content", {0, 0}, ImGuiChildFlags_AlwaysUseWindowPadding)) {
        if (_scrollToTop) {
            ImGui::SetScrollY(0.0f);
            _scrollToTop = false;
        }
        if (_chapters.empty()) {
            AlienGui::Text("No documentation found in " + Const::DocsPath.string() + ".");
        } else {
            auto clickedLink = _renderer.render(_chapters.at(_currentChapterIndex).document, Const::DocsPath);
            processChapterButtons();
            if (clickedLink.has_value()) {
                openLink(*clickedLink);
            }
        }
    }
    ImGui::EndChild();
}

void DocumentationWindow::processChapterButtons()
{
    auto chapterIndex = _currentChapterIndex;
    auto hasPreviousChapter = chapterIndex > 0;
    auto hasNextChapter = chapterIndex + 1 < _chapters.size();
    if (!hasPreviousChapter && !hasNextChapter) {
        return;
    }

    AlienGui::Separator();
    if (hasPreviousChapter) {
        if (AlienGui::Button(ICON_FA_CHEVRON_LEFT "  " + _chapters.at(chapterIndex - 1).title)) {
            openChapter(chapterIndex - 1);
        }
    }
    if (hasNextChapter) {
        auto text = _chapters.at(chapterIndex + 1).title + "  " ICON_FA_CHEVRON_RIGHT;
        auto buttonWidth = ImGui::CalcTextSize(text.c_str()).x + ImGui::GetStyle().FramePadding.x * 2;
        if (hasPreviousChapter) {
            ImGui::SameLine();
        }
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - buttonWidth));
        if (AlienGui::Button(text)) {
            openChapter(chapterIndex + 1);
        }
    }
    ImGui::Dummy({0.0f, ImGui::GetStyle().ItemSpacing.y});
}

void DocumentationWindow::openLink(std::string const& link)
{
    if (link.starts_with("http://") || link.starts_with("https://")) {
        WebLinkHelper::openInBrowser(link);
        return;
    }
    auto separatorPos = link.find('#');
    auto filename = link.substr(0, separatorPos);
    auto anchor = separatorPos != std::string::npos ? std::make_optional(link.substr(separatorPos + 1)) : std::nullopt;

    if (filename.empty()) {
        if (anchor.has_value()) {
            _renderer.scrollToAnchor(*anchor);
        }
        return;
    }
    auto findResult = std::ranges::find(_chapters, filename, &Chapter::filename);
    if (findResult != _chapters.end()) {
        openChapter(std::distance(_chapters.begin(), findResult), anchor);
    }
}

void DocumentationWindow::openChapter(size_t chapterIndex, std::optional<std::string> const& anchor)
{
    // A found anchor overrides the scroll position in the same frame
    if (chapterIndex != _currentChapterIndex || !anchor.has_value()) {
        _scrollToTop = true;
    }
    _currentChapterIndex = chapterIndex;
    if (anchor.has_value()) {
        _renderer.scrollToAnchor(*anchor);
    }
}
