#include <gtest/gtest.h>

#include <Base/DocumentationReader.h>

class DocumentationReaderTests : public ::testing::Test
{};

TEST_F(DocumentationReaderTests, parseChapters_emptyText)
{
    EXPECT_TRUE(DocumentationReader::parseChapters("").empty());
}

TEST_F(DocumentationReaderTests, parseChapters_parts)
{
    auto chapters = DocumentationReader::parseChapters("Introduction with a [link](format-reference.md).\n\n"
                                                       "# Start here\n\n"
                                                       "- [Getting started](getting-started.md)\n\n"
                                                       "# Reference\n\n"
                                                       "- [Cell types](cell-types.md)\n"
                                                       "- Item without link\n"
                                                       "- [Energy](energy.md)\n");

    std::vector<DocumentationChapter> expectedChapters = {
        DocumentationChapter{.filename = "getting-started.md", .title = "Getting started", .part = "Start here"},
        DocumentationChapter{.filename = "cell-types.md", .title = "Cell types", .part = "Reference"},
        DocumentationChapter{.filename = "energy.md", .title = "Energy", .part = "Reference"},
    };
    EXPECT_EQ(expectedChapters, chapters);
}
