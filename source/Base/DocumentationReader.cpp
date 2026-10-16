#include "DocumentationReader.h"

#include <algorithm>
#include <fstream>
#include <iterator>

#include "MarkdownParser.h"
#include "Resources.h"

namespace
{
    auto constexpr ContentsFilename = "contents.md";

    std::string readFile(std::filesystem::path const& path)
    {
        std::ifstream stream(path);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }
}

std::vector<DocumentationChapter> DocumentationReader::getChapters()
{
    return parseChapters(readFile(Const::DocsPath / ContentsFilename));
}

std::vector<DocumentationChapter> DocumentationReader::parseChapters(std::string const& contentsText)
{
    // The table of contents lists the chapters as links, grouped under part headings
    std::vector<DocumentationChapter> result;
    std::string part;
    for (auto const& block : MarkdownParser::parse(contentsText).blocks) {
        if (auto heading = std::get_if<MarkdownHeading>(&block)) {
            part = MarkdownParser::toPlainText(heading->spans);
            continue;
        }
        auto listItem = std::get_if<MarkdownListItem>(&block);
        if (!listItem) {
            continue;
        }
        auto linkSpan = std::ranges::find_if(listItem->spans, [](auto const& span) { return span.link.has_value(); });
        if (linkSpan == listItem->spans.end()) {
            continue;
        }
        result.emplace_back(DocumentationChapter{.filename = *linkSpan->link, .title = linkSpan->text, .part = part});
    }
    return result;
}

std::string DocumentationReader::readChapter(DocumentationChapter const& chapter)
{
    return readFile(Const::DocsPath / chapter.filename);
}
