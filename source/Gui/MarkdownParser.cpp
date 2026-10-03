#include "MarkdownParser.h"

#include <unordered_map>
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

    enum class Collector
    {
        None,
        Paragraph,
        Heading,
        ListItem,
        TableCell,
        CodeBlock
    };

    struct ListLevel
    {
        bool ordered = false;
        int nextNumber = 1;
    };

    struct ParserState
    {
        MarkdownDocument document;

        Collector collector = Collector::None;
        MarkdownSpans spans;
        std::string codeText;
        int headingLevel = 1;

        std::vector<ListLevel> lists;
        std::optional<MarkdownListItem> listItem;

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

        void flushParagraph()
        {
            if (spans.empty()) {
                return;
            }
            if (quoteDepth > 0) {
                note.paragraphs.emplace_back(std::move(spans));
            } else {
                document.blocks.emplace_back(MarkdownParagraph{.spans = std::move(spans)});
            }
            spans.clear();
        }

        void flushListItem()
        {
            if (listItem.has_value() && !spans.empty()) {
                listItem->spans = std::move(spans);
                document.blocks.emplace_back(std::move(*listItem));
            }
            listItem.reset();
            spans.clear();
        }
    };

    int enterBlock(MD_BLOCKTYPE type, void* detail, void* userData)
    {
        auto& state = *static_cast<ParserState*>(userData);
        switch (type) {
        case MD_BLOCK_QUOTE:
            if (state.quoteDepth++ == 0) {
                state.note = MarkdownNote();
            }
            break;
        case MD_BLOCK_UL:
            state.flushListItem();
            state.lists.emplace_back(ListLevel{.ordered = false});
            break;
        case MD_BLOCK_OL:
            state.flushListItem();
            state.lists.emplace_back(ListLevel{.ordered = true, .nextNumber = static_cast<int>(static_cast<MD_BLOCK_OL_DETAIL*>(detail)->start)});
            break;
        case MD_BLOCK_LI: {
            state.flushListItem();
            auto& list = state.lists.back();
            state.listItem = MarkdownListItem{.depth = static_cast<int>(state.lists.size()) - 1};
            if (list.ordered) {
                state.listItem->number = list.nextNumber++;
            }
            state.collector = Collector::ListItem;
        } break;
        case MD_BLOCK_HR:
            state.document.blocks.emplace_back(MarkdownRule());
            break;
        case MD_BLOCK_H:
            state.headingLevel = static_cast<int>(static_cast<MD_BLOCK_H_DETAIL*>(detail)->level);
            state.spans.clear();
            state.collector = Collector::Heading;
            break;
        case MD_BLOCK_CODE:
            state.codeText.clear();
            state.collector = Collector::CodeBlock;
            break;
        case MD_BLOCK_P:
            if (state.collector == Collector::ListItem) {
                if (!state.spans.empty()) {
                    state.appendText(" ");
                }
            } else {
                state.spans.clear();
                state.collector = Collector::Paragraph;
            }
            break;
        case MD_BLOCK_TABLE:
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
            state.collector = Collector::TableCell;
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
            state.flushParagraph();
            if (--state.quoteDepth == 0 && !state.note.paragraphs.empty()) {
                state.document.blocks.emplace_back(std::move(state.note));
            }
            break;
        case MD_BLOCK_UL:
        case MD_BLOCK_OL:
            state.flushListItem();
            state.lists.pop_back();
            state.collector = state.lists.empty() ? Collector::None : Collector::ListItem;
            break;
        case MD_BLOCK_LI:
            state.flushListItem();
            break;
        case MD_BLOCK_H: {
            auto anchor = MarkdownParser::toAnchor(MarkdownParser::toPlainText(state.spans));
            state.document.blocks.emplace_back(MarkdownHeading{.level = state.headingLevel, .spans = std::move(state.spans), .anchor = std::move(anchor)});
            state.spans.clear();
            state.collector = Collector::None;
        } break;
        case MD_BLOCK_CODE:
            while (!state.codeText.empty() && state.codeText.back() == '\n') {
                state.codeText.pop_back();
            }
            state.document.blocks.emplace_back(MarkdownCodeBlock{.text = std::move(state.codeText)});
            state.codeText.clear();
            state.collector = Collector::None;
            break;
        case MD_BLOCK_P:
            if (state.collector == Collector::Paragraph) {
                state.flushParagraph();
                state.collector = Collector::None;
            }
            break;
        case MD_BLOCK_TH:
        case MD_BLOCK_TD:
            state.tableRow.emplace_back(std::move(state.spans));
            state.spans.clear();
            state.collector = Collector::None;
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
            state.document.blocks.emplace_back(std::move(state.table));
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
            if (state.collector == Collector::Paragraph && state.quoteDepth == 0) {
                state.flushParagraph();
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
                state.document.blocks.emplace_back(std::move(*state.image));
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
        if (state.collector == Collector::CodeBlock) {
            state.codeText += string;
            return 0;
        }
        if (state.collector == Collector::None) {
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
