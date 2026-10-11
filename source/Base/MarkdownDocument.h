#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

struct MarkdownSpan
{
    bool operator==(MarkdownSpan const&) const = default;

    std::string text;
    bool bold = false;
    bool italic = false;
    bool code = false;
    std::optional<std::string> link;
};
using MarkdownSpans = std::vector<MarkdownSpan>;

struct MarkdownHeading
{
    bool operator==(MarkdownHeading const&) const = default;

    int level = 1;
    MarkdownSpans spans;
    std::string anchor;
};

struct MarkdownParagraph
{
    bool operator==(MarkdownParagraph const&) const = default;

    MarkdownSpans spans;
    int listIndentLevel = 0;
};

struct MarkdownListItem
{
    bool operator==(MarkdownListItem const&) const = default;

    int depth = 0;
    std::optional<int> number;
    MarkdownSpans spans;
};

struct MarkdownImage
{
    bool operator==(MarkdownImage const&) const = default;

    std::string source;
    std::string caption;
};

struct MarkdownCodeBlock
{
    bool operator==(MarkdownCodeBlock const&) const = default;

    std::string text;
    int listIndentLevel = 0;
};

using MarkdownNoteBlock = std::variant<MarkdownParagraph, MarkdownListItem, MarkdownCodeBlock>;

struct MarkdownNote
{
    bool operator==(MarkdownNote const&) const = default;

    std::vector<MarkdownNoteBlock> blocks;
};

enum class MarkdownAlignment
{
    Left,
    Center,
    Right
};

struct MarkdownTable
{
    bool operator==(MarkdownTable const&) const = default;

    std::vector<MarkdownSpans> header;
    std::vector<MarkdownAlignment> alignments;
    std::vector<std::vector<MarkdownSpans>> rows;
};

struct MarkdownRule
{
    bool operator==(MarkdownRule const&) const = default;
};

using MarkdownBlock =
    std::variant<MarkdownHeading, MarkdownParagraph, MarkdownListItem, MarkdownImage, MarkdownCodeBlock, MarkdownNote, MarkdownTable, MarkdownRule>;

struct MarkdownDocument
{
    bool operator==(MarkdownDocument const&) const = default;

    std::vector<MarkdownBlock> blocks;
};
