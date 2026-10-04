#include "MarkdownParser.h"

#include <unordered_map>
#include <utility>
#include <cctype>

#include <md4c.h>

namespace
{
    std::string toString(MD_ATTRIBUTE const& attribute)
    {
        return {attribute.text, attribute.size};
    }

    std::string decodeEntity(std::string const& entity)
    {
        static std::unordered_map<std::string, std::string> const DecodedEntities = {
            {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&nbsp;", " "}};
        auto findResult = DecodedEntities.find(entity);
        return findResult != DecodedEntities.end() ? findResult->second : entity;
    }

    MarkdownAlignment toAlignment(MD_ALIGN align)
    {
        switch (align) {
        case MD_ALIGN_CENTER:
            return MarkdownAlignment::Center;
        case MD_ALIGN_RIGHT:
            return MarkdownAlignment::Right;
        default:
            return MarkdownAlignment::Left;
        }
    }

    enum class Leaf
    {
        None,
        Paragraph,
        Heading,
        TableCell,
        CodeBlock
    };

    enum class Container
    {
        ListItem,
        Quote
    };

    struct ListLevel
    {
        bool ordered = false;
        int nextNumber = 1;
    };

    struct OpenListItem
    {
        int depth = 0;
        std::optional<int> number;
        bool markerAdded = false;
    };

    struct ParserState
    {
        MarkdownDocument document;

        Leaf leaf = Leaf::None;
        MarkdownSpans spans;
        std::string codeText;
        int headingLevel = 1;
        std::unordered_map<std::string, int> numHeadingsByAnchor;

        std::vector<Container> containers;
        std::vector<ListLevel> lists;
        std::vector<OpenListItem> listItems;
        int quoteDepth = 0;
        MarkdownNote note;

        MarkdownTable table;
        bool inTableHead = false;
        std::vector<MarkdownSpans> tableRow;

        int boldDepth = 0;
        int italicDepth = 0;
        int codeDepth = 0;
        std::vector<std::string> links;
        std::optional<MarkdownImage> image;

        bool isInListItem() const { return !containers.empty() && containers.back() == Container::ListItem; }

        int getListIndentLevel() const { return isInListItem() ? static_cast<int>(listItems.size()) : 0; }

        void appendText(std::string const& text)
        {
            MarkdownSpan span{.bold = boldDepth > 0, .italic = italicDepth > 0, .code = codeDepth > 0};
            if (!links.empty()) {
                span.link = links.back();
            }
            if (!spans.empty()) {
                auto& lastSpan = spans.back();
                if (lastSpan.bold == span.bold && lastSpan.italic == span.italic && lastSpan.code == span.code && lastSpan.link == span.link) {
                    lastSpan.text += text;
                    return;
                }
            }
            span.text = text;
            spans.emplace_back(std::move(span));
        }

        // Paragraphs, list items and code blocks inside a block quote belong to its note
        void addBlock(MarkdownNoteBlock&& block)
        {
            if (quoteDepth > 0) {
                note.blocks.emplace_back(std::move(block));
            } else {
                std::visit([this](auto&& value) { document.blocks.emplace_back(std::move(value)); }, std::move(block));
            }
        }

        // Other blocks interrupt a note, the rest of the block quote continues in a new note
        void addStandaloneBlock(MarkdownBlock&& block)
        {
            flushNote();
            document.blocks.emplace_back(std::move(block));
        }

        void flushNote()
        {
            if (!note.blocks.empty()) {
                document.blocks.emplace_back(std::move(note));
            }
            note = MarkdownNote();
        }

        // The first text of a list item carries the marker, further text of the item is indented below it
        void flushText()
        {
            if (spans.empty()) {
                return;
            }
            if (isInListItem() && !listItems.back().markerAdded) {
                auto& listItem = listItems.back();
                addBlock(MarkdownListItem{.depth = listItem.depth, .number = listItem.number, .spans = std::move(spans)});
                listItem.markerAdded = true;
            } else {
                addBlock(MarkdownParagraph{.spans = std::move(spans), .listIndentLevel = getListIndentLevel()});
            }
            spans.clear();
        }

        std::string createUniqueAnchor(std::string const& headingText)
        {
            auto anchor = MarkdownParser::toAnchor(headingText);
            auto numPreviousHeadings = numHeadingsByAnchor[anchor]++;
            return numPreviousHeadings == 0 ? anchor : anchor + "-" + std::to_string(numPreviousHeadings);
        }
    };

    int enterBlock(MD_BLOCKTYPE type, void* detail, void* userData)
    {
        auto& state = *static_cast<ParserState*>(userData);
        switch (type) {
        case MD_BLOCK_QUOTE:
            state.flushText();
            state.containers.emplace_back(Container::Quote);
            ++state.quoteDepth;
            break;
        case MD_BLOCK_UL:
            state.flushText();
            state.lists.emplace_back(ListLevel{.ordered = false});
            break;
        case MD_BLOCK_OL:
            state.flushText();
            state.lists.emplace_back(ListLevel{.ordered = true, .nextNumber = static_cast<int>(static_cast<MD_BLOCK_OL_DETAIL*>(detail)->start)});
            break;
        case MD_BLOCK_LI: {
            auto& list = state.lists.back();
            auto number = list.ordered ? std::make_optional(list.nextNumber++) : std::nullopt;
            state.listItems.emplace_back(OpenListItem{.depth = static_cast<int>(state.lists.size()) - 1, .number = number});
            state.containers.emplace_back(Container::ListItem);
        } break;
        case MD_BLOCK_HR:
            state.flushText();
            state.addStandaloneBlock(MarkdownRule());
            break;
        case MD_BLOCK_H:
            state.flushText();
            state.headingLevel = static_cast<int>(static_cast<MD_BLOCK_H_DETAIL*>(detail)->level);
            state.leaf = Leaf::Heading;
            break;
        case MD_BLOCK_CODE:
            state.flushText();
            state.codeText.clear();
            state.leaf = Leaf::CodeBlock;
            break;
        case MD_BLOCK_P:
            state.flushText();
            state.leaf = Leaf::Paragraph;
            break;
        case MD_BLOCK_TABLE:
            state.flushText();
            state.table = MarkdownTable();
            break;
        case MD_BLOCK_THEAD:
            state.inTableHead = true;
            break;
        case MD_BLOCK_TBODY:
            state.inTableHead = false;
            break;
        case MD_BLOCK_TR:
            state.tableRow.clear();
            break;
        case MD_BLOCK_TH:
        case MD_BLOCK_TD:
            if (type == MD_BLOCK_TH) {
                state.table.alignments.emplace_back(toAlignment(static_cast<MD_BLOCK_TD_DETAIL*>(detail)->align));
            }
            state.spans.clear();
            state.leaf = Leaf::TableCell;
            break;
        default:
            break;
        }
        return 0;
    }

    int leaveBlock(MD_BLOCKTYPE type, void*, void* userData)
    {
        auto& state = *static_cast<ParserState*>(userData);
        switch (type) {
        case MD_BLOCK_QUOTE:
            state.flushText();
            state.containers.pop_back();
            if (--state.quoteDepth == 0) {
                state.flushNote();
            }
            break;
        case MD_BLOCK_UL:
        case MD_BLOCK_OL:
            state.lists.pop_back();
            break;
        case MD_BLOCK_LI:
            state.flushText();
            state.listItems.pop_back();
            state.containers.pop_back();
            break;
        case MD_BLOCK_H: {
            auto anchor = state.createUniqueAnchor(MarkdownParser::toPlainText(state.spans));
            state.addStandaloneBlock(MarkdownHeading{.level = state.headingLevel, .spans = std::move(state.spans), .anchor = std::move(anchor)});
            state.spans.clear();
            state.leaf = Leaf::None;
        } break;
        case MD_BLOCK_CODE:
            while (!state.codeText.empty() && state.codeText.back() == '\n') {
                state.codeText.pop_back();
            }
            state.addBlock(MarkdownCodeBlock{.text = std::move(state.codeText), .listIndentLevel = state.getListIndentLevel()});
            state.codeText.clear();
            state.leaf = Leaf::None;
            break;
        case MD_BLOCK_P:
            state.flushText();
            state.leaf = Leaf::None;
            break;
        case MD_BLOCK_TH:
        case MD_BLOCK_TD:
            state.tableRow.emplace_back(std::move(state.spans));
            state.spans.clear();
            state.leaf = Leaf::None;
            break;
        case MD_BLOCK_TR:
            if (state.inTableHead) {
                state.table.header = std::move(state.tableRow);
            } else {
                state.table.rows.emplace_back(std::move(state.tableRow));
            }
            state.tableRow.clear();
            break;
        case MD_BLOCK_TABLE:
            state.addStandaloneBlock(std::move(state.table));
            break;
        default:
            break;
        }
        return 0;
    }

    int enterSpan(MD_SPANTYPE type, void* detail, void* userData)
    {
        auto& state = *static_cast<ParserState*>(userData);
        switch (type) {
        case MD_SPAN_EM:
            ++state.italicDepth;
            break;
        case MD_SPAN_STRONG:
            ++state.boldDepth;
            break;
        case MD_SPAN_CODE:
            ++state.codeDepth;
            break;
        case MD_SPAN_A:
            state.links.emplace_back(toString(static_cast<MD_SPAN_A_DETAIL*>(detail)->href));
            break;
        case MD_SPAN_IMG:
            // Images inside lists, tables or notes are not supported and fall back to their caption text
            if (state.leaf == Leaf::Paragraph && state.containers.empty()) {
                state.flushText();
                state.image = MarkdownImage{.source = toString(static_cast<MD_SPAN_IMG_DETAIL*>(detail)->src)};
            }
            break;
        default:
            break;
        }
        return 0;
    }

    int leaveSpan(MD_SPANTYPE type, void*, void* userData)
    {
        auto& state = *static_cast<ParserState*>(userData);
        switch (type) {
        case MD_SPAN_EM:
            --state.italicDepth;
            break;
        case MD_SPAN_STRONG:
            --state.boldDepth;
            break;
        case MD_SPAN_CODE:
            --state.codeDepth;
            break;
        case MD_SPAN_A:
            state.links.pop_back();
            break;
        case MD_SPAN_IMG:
            if (state.image.has_value()) {
                state.addStandaloneBlock(std::move(*state.image));
                state.image.reset();
            }
            break;
        default:
            break;
        }
        return 0;
    }

    int processText(MD_TEXTTYPE type, MD_CHAR const* text, MD_SIZE size, void* userData)
    {
        auto& state = *static_cast<ParserState*>(userData);
        auto string = std::string(text, size);
        if (type == MD_TEXT_ENTITY) {
            string = decodeEntity(string);
        }
        if (state.image.has_value()) {
            state.image->caption += string;
            return 0;
        }
        if (state.leaf == Leaf::CodeBlock) {
            state.codeText += string;
            return 0;
        }
        // The text of tight list items is not wrapped in paragraphs
        if (state.leaf == Leaf::None && !state.isInListItem()) {
            return 0;
        }
        switch (type) {
        case MD_TEXT_NORMAL:
        case MD_TEXT_CODE:
        case MD_TEXT_ENTITY:
            state.appendText(string);
            break;
        case MD_TEXT_SOFTBR:
            state.appendText(" ");
            break;
        case MD_TEXT_BR:
            state.appendText("\n");
            break;
        default:
            break;
        }
        return 0;
    }
}

MarkdownDocument MarkdownParser::parse(std::string const& text)
{
    MD_PARSER parser{
        .abi_version = 0,
        .flags = MD_DIALECT_GITHUB | MD_FLAG_NOHTML,
        .enter_block = enterBlock,
        .leave_block = leaveBlock,
        .enter_span = enterSpan,
        .leave_span = leaveSpan,
        .text = processText,
        .debug_log = nullptr,
        .syntax = nullptr};
    ParserState state;
    md_parse(text.data(), static_cast<MD_SIZE>(text.size()), &parser, &state);
    return std::move(state.document);
}

std::string MarkdownParser::toPlainText(MarkdownSpans const& spans)
{
    std::string result;
    for (auto const& span : spans) {
        result += span.text;
    }
    return result;
}

std::string MarkdownParser::toAnchor(std::string const& headingText)
{
    std::string result;
    for (auto c : headingText) {
        auto uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '-' || c == '_' || uc >= 0x80) {
            result += static_cast<char>(std::tolower(uc));
        } else if (c == ' ') {
            result += '-';
        }
    }
    return result;
}
