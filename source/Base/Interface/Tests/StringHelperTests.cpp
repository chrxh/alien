#include <gtest/gtest.h>

#include <Base/Interface/StringHelper.h>

class StringHelperTests : public ::testing::Test
{};

TEST_F(StringHelperTests, join)
{
    EXPECT_EQ("", StringHelper::join({}));
    EXPECT_EQ("7", StringHelper::join({7}));
    EXPECT_EQ("1, 4, 0", StringHelper::join({1, 4, 0}));
}

TEST_F(StringHelperTests, formatRanges)
{
    EXPECT_EQ("", StringHelper::formatRanges({}));
    EXPECT_EQ("3", StringHelper::formatRanges({3}));
    EXPECT_EQ("0-2, 5", StringHelper::formatRanges({0, 1, 2, 5}));
    EXPECT_EQ("1, 3-4, 9-11", StringHelper::formatRanges({1, 3, 4, 9, 10, 11}));
}

TEST_F(StringHelperTests, formatEnumeration)
{
    EXPECT_EQ("", StringHelper::formatEnumeration({}));
    EXPECT_EQ("a", StringHelper::formatEnumeration({"a"}));
    EXPECT_EQ("a and b", StringHelper::formatEnumeration({"a", "b"}));
    EXPECT_EQ("a, b and c", StringHelper::formatEnumeration({"a", "b", "c"}));
}
