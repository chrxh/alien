#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

struct MarkdownSpan
{
    std::string text;
    bool bold = false;
    bool italic = false;
    bool code = false;
    std::optional<std::string> link;
};
using MarkdownSpans = std::vector<MarkdownSpan>;

struct MarkdownHeading
{
    int level = 1;
    MarkdownSpans spans;
    std::string anchor;
};

struct MarkdownParagraph
{
    MarkdownSpans spans;
};

struct MarkdownListItem
{
    int depth = 0;
    std::optional<int> number;
    MarkdownSpans spans;
};

struct MarkdownImage
{
    std::string source;
    std::string caption;
};

struct MarkdownCodeBlock
{
    std::string text;
};

struct MarkdownNote
{
    std::vector<MarkdownSpans> paragraphs;
};

enum class MarkdownAlignment
{
    Left,
    Center,
    Right
};

struct MarkdownTable
{
    std::vector<MarkdownSpans> header;
    std::vector<MarkdownAlignment> alignments;
    std::vector<std::vector<MarkdownSpans>> rows;
};

struct MarkdownRule
{};

using MarkdownBlock =
    std::variant<MarkdownHeading, MarkdownParagraph, MarkdownListItem, MarkdownImage, MarkdownCodeBlock, MarkdownNote, MarkdownTable, MarkdownRule>;

struct MarkdownDocument
{
    std::vector<MarkdownBlock> blocks;
};
