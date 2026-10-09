#include <gtest/gtest.h>

#include <Base/MarkdownParser.h>

class MarkdownParserTests : public ::testing::Test
{
protected:
    MarkdownSpans toSpans(std::string const& text) const { return {MarkdownSpan{.text = text}}; }
};

TEST_F(MarkdownParserTests, parse_emptyText)
{
    auto document = MarkdownParser::parse("");

    EXPECT_TRUE(document.blocks.empty());
}

TEST_F(MarkdownParserTests, parse_headings)
{
    auto document = MarkdownParser::parse("# Cell types\n\n## How it works\n\n### Details\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownHeading{.level = 1, .spans = toSpans("Cell types"), .anchor = "cell-types"},
        MarkdownHeading{.level = 2, .spans = toSpans("How it works"), .anchor = "how-it-works"},
        MarkdownHeading{.level = 3, .spans = toSpans("Details"), .anchor = "details"},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_repeatedHeadings)
{
    auto document = MarkdownParser::parse("## Signals\n\n## Modes\n\n## Signals\n\n## Signals\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownHeading{.level = 2, .spans = toSpans("Signals"), .anchor = "signals"},
        MarkdownHeading{.level = 2, .spans = toSpans("Modes"), .anchor = "modes"},
        MarkdownHeading{.level = 2, .spans = toSpans("Signals"), .anchor = "signals-1"},
        MarkdownHeading{.level = 2, .spans = toSpans("Signals"), .anchor = "signals-2"},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_inlineStyles)
{
    auto document = MarkdownParser::parse("Text with **bold**, *italic*, `code` and [a link](target.md#anchor).\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownParagraph{
        .spans = {
            {.text = "Text with "},
            {.text = "bold", .bold = true},
            {.text = ", "},
            {.text = "italic", .italic = true},
            {.text = ", "},
            {.text = "code", .code = true},
            {.text = " and "},
            {.text = "a link", .link = "target.md#anchor"},
            {.text = "."},
        }}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_bareUrl)
{
    auto document = MarkdownParser::parse("Please visit https://github.com/chrxh/alien for updates.\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownParagraph{
        .spans = {
            {.text = "Please visit "},
            {.text = "https://github.com/chrxh/alien", .link = "https://github.com/chrxh/alien"},
            {.text = " for updates."},
        }}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_lineBreaks)
{
    auto document = MarkdownParser::parse("First line\nsame paragraph\\\nnew line\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownParagraph{.spans = toSpans("First line same paragraph\nnew line")}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_entities)
{
    auto document = MarkdownParser::parse("Fish &amp; chips &lt;3\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownParagraph{.spans = toSpans("Fish & chips <3")}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_lists)
{
    auto document = MarkdownParser::parse("- First\n  - Nested\n- Second\n\n3. Third\n4. Fourth\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownListItem{.depth = 0, .spans = toSpans("First")},
        MarkdownListItem{.depth = 1, .spans = toSpans("Nested")},
        MarkdownListItem{.depth = 0, .spans = toSpans("Second")},
        MarkdownListItem{.depth = 0, .number = 3, .spans = toSpans("Third")},
        MarkdownListItem{.depth = 0, .number = 4, .spans = toSpans("Fourth")},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_listItemWithCodeBlockAndParagraph)
{
    auto document = MarkdownParser::parse("1. Step one\n\n   ```\n   code\n   ```\n\n   More text\n\n2. Step two\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownListItem{.depth = 0, .number = 1, .spans = toSpans("Step one")},
        MarkdownCodeBlock{.text = "code", .listIndentLevel = 1},
        MarkdownParagraph{.spans = toSpans("More text"), .listIndentLevel = 1},
        MarkdownListItem{.depth = 0, .number = 2, .spans = toSpans("Step two")},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_listItemContinuedAfterNestedList)
{
    auto document = MarkdownParser::parse("- Parent\n  - Child\n\n  More parent text\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownListItem{.depth = 0, .spans = toSpans("Parent")},
        MarkdownListItem{.depth = 1, .spans = toSpans("Child")},
        MarkdownParagraph{.spans = toSpans("More parent text"), .listIndentLevel = 1},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_noteWithList)
{
    auto document = MarkdownParser::parse("> Tip:\n> - First\n> - Second\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownNote{
        .blocks = {
            MarkdownParagraph{.spans = toSpans("Tip:")},
            MarkdownListItem{.depth = 0, .spans = toSpans("First")},
            MarkdownListItem{.depth = 0, .spans = toSpans("Second")},
        }}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_noteInterruptedByHeading)
{
    auto document = MarkdownParser::parse("> First\n>\n> ## Heading\n>\n> Second\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownNote{.blocks = {MarkdownParagraph{.spans = toSpans("First")}}},
        MarkdownHeading{.level = 2, .spans = toSpans("Heading"), .anchor = "heading"},
        MarkdownNote{.blocks = {MarkdownParagraph{.spans = toSpans("Second")}}},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_image)
{
    auto document = MarkdownParser::parse("Before\n\n![The caption](images/picture.png)\n\nAfter\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownParagraph{.spans = toSpans("Before")},
        MarkdownImage{.source = "images/picture.png", .caption = "The caption"},
        MarkdownParagraph{.spans = toSpans("After")},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_imageInListItem)
{
    auto document = MarkdownParser::parse("- ![The caption](images/picture.png)\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownListItem{.depth = 0, .spans = toSpans("The caption")}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_table)
{
    auto document = MarkdownParser::parse("| Name | Value | Note |\n| --- | ---: | :---: |\n| **A** | 1 | x |\n| B | 2 | y |\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownTable{
        .header = {toSpans("Name"), toSpans("Value"), toSpans("Note")},
        .alignments = {MarkdownAlignment::Left, MarkdownAlignment::Right, MarkdownAlignment::Center},
        .rows = {
            {MarkdownSpans{{.text = "A", .bold = true}}, toSpans("1"), toSpans("x")},
            {toSpans("B"), toSpans("2"), toSpans("y")},
        }}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_codeBlock)
{
    auto document = MarkdownParser::parse("```\nfirst line\nsecond line\n```\n");

    std::vector<MarkdownBlock> expectedBlocks = {MarkdownCodeBlock{.text = "first line\nsecond line"}};
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, parse_rule)
{
    auto document = MarkdownParser::parse("Above\n\n---\n\nBelow\n");

    std::vector<MarkdownBlock> expectedBlocks = {
        MarkdownParagraph{.spans = toSpans("Above")},
        MarkdownRule(),
        MarkdownParagraph{.spans = toSpans("Below")},
    };
    EXPECT_EQ(expectedBlocks, document.blocks);
}

TEST_F(MarkdownParserTests, toAnchor)
{
    EXPECT_EQ("cell-types", MarkdownParser::toAnchor("Cell types"));
    EXPECT_EQ("whats-new-in-50", MarkdownParser::toAnchor("What's new in 5.0?"));
    EXPECT_EQ("snake_case-and-dashes", MarkdownParser::toAnchor("Snake_case and-dashes"));
    EXPECT_EQ("gr\xC3\xB6\xC3\x9Fte-zahl", MarkdownParser::toAnchor("Gr\xC3\xB6\xC3\x9Fte Zahl"));
}

TEST_F(MarkdownParserTests, toPlainText)
{
    MarkdownSpans spans = {{.text = "Plain "}, {.text = "bold", .bold = true}, {.text = " text"}};

    EXPECT_EQ("Plain bold text", MarkdownParser::toPlainText(spans));
}
