#include "McpDocumentationTools.h"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/DocumentationReader.h>
#include <Base/StringHelper.h>

#include <Network/McpArguments.h>
#include <Network/McpJson.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr ChapterExtension = ".md";
}

std::vector<McpTool> McpDocumentationTools::getTools()
{
    return {
        McpTool{
            .name = "read_documentation",
            .description = "Reads the documentation of ALIEN. It explains in detail how creatures live, the cell types, genomes, neural networks, energy "
                           "and all simulation parameters. Without a chapter, the table of contents is returned. With a chapter, its text is returned "
                           "in Markdown. A link to another chapter such as 'energy.md#the-external-energy-pool' refers to the chapter 'energy.md'.",
            .inputSchema = McpSchema::object({{"chapter", McpSchema::string("File name of the chapter from the table of contents, e.g. 'cell-types.md'")}}),
            .handler = [this](boost::json::object const& arguments) { return readDocumentation(arguments); },
        },
    };
}

McpToolResult McpDocumentationTools::readDocumentation(boost::json::object const& arguments) const
{
    auto chapters = DocumentationReader::getChapters();

    auto chapterArgument = McpArguments::getOptionalString(arguments, "chapter");
    if (!chapterArgument.has_value()) {
        boost::json::array contents;
        for (auto const& chapter : chapters) {
            contents.emplace_back(boost::json::object{{"chapter", chapter.filename}, {"title", chapter.title}, {"part", chapter.part}});
        }
        return {.text = McpJson::serialize(boost::json::object{{"chapters", std::move(contents)}})};
    }

    auto filename = chapterArgument->substr(0, chapterArgument->find('#'));
    if (!filename.ends_with(ChapterExtension)) {
        filename += ChapterExtension;
    }
    auto chapter = std::ranges::find(chapters, filename, &DocumentationChapter::filename);
    if (chapter == chapters.end()) {
        auto filenames = chapters | std::views::transform(&DocumentationChapter::filename) | std::ranges::to<std::vector>();
        throw std::invalid_argument(std::format("Unknown chapter '{}'. Available chapters: {}.", *chapterArgument, StringHelper::formatEnumeration(filenames)));
    }
    auto text = DocumentationReader::readChapter(*chapter);
    if (text.empty()) {
        throw std::runtime_error(std::format("The chapter '{}' could not be read.", chapter->filename));
    }
    return {.text = text};
}
