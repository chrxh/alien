#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Base/StringHelper.h>

class StringHelperTests : public ::testing::Test
{};

TEST_F(StringHelperTests, join)
{
    EXPECT_EQ("", StringHelper::join({}));
    EXPECT_EQ("7", StringHelper::join({7}));
    EXPECT_EQ("1, 4, 0", StringHelper::join({1, 4, 0}));
}

TEST_F(StringHelperTests, formatUtc)
{
    auto day = std::chrono::sys_days(std::chrono::year(2026) / std::chrono::October / std::chrono::day(7));
    EXPECT_EQ("2026-10-07 00:00:00", StringHelper::formatUtc(day));
    EXPECT_EQ("2026-10-07 18:19:05", StringHelper::formatUtc(day + std::chrono::hours(18) + std::chrono::minutes(19) + std::chrono::milliseconds(5900)));
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

TEST_F(StringHelperTests, formatCompact)
{
    EXPECT_EQ("0", StringHelper::formatCompact(0));
    EXPECT_EQ("999", StringHelper::formatCompact(999));
    EXPECT_EQ("1K", StringHelper::formatCompact(1000));
    EXPECT_EQ("99.9K", StringHelper::formatCompact(99940));
    EXPECT_EQ("100K", StringHelper::formatCompact(99960));
    EXPECT_EQ("125K", StringHelper::formatCompact(125000));
    EXPECT_EQ("1M", StringHelper::formatCompact(999600));
    EXPECT_EQ("11.4M", StringHelper::formatCompact(11380000));
    EXPECT_EQ("2.1B", StringHelper::formatCompact(2100000000));
}

TEST_F(StringHelperTests, formatFileSize)
{
    EXPECT_EQ("900 B", StringHelper::formatFileSize(900));
    EXPECT_EQ("2 KB", StringHelper::formatFileSize(2048));
    EXPECT_EQ("1.5 KB", StringHelper::formatFileSize(1536));
    EXPECT_EQ("318 MB", StringHelper::formatFileSize(333447168));
    EXPECT_EQ("1 GB", StringHelper::formatFileSize(1073741000));
    EXPECT_EQ("6.1 GB", StringHelper::formatFileSize(6549825126));
}

TEST_F(StringHelperTests, formatCount)
{
    EXPECT_EQ("0 states", StringHelper::formatCount(0, "state"));
    EXPECT_EQ("1 state", StringHelper::formatCount(1, "state"));
    EXPECT_EQ("1,200 save points", StringHelper::formatCount(1200, "save point"));
}

TEST_F(StringHelperTests, pathConversion)
{
    std::string name = "J\xc3\xbcrgen/Ocean temple";
    auto path = StringHelper::toPath(name);
#ifdef _WIN32
    EXPECT_EQ(std::wstring(L"J\x00fcrgen/Ocean temple"), path.wstring());
#endif
    EXPECT_EQ(name, StringHelper::toUtf8(path));
}

TEST_F(StringHelperTests, naturalLess)
{
    EXPECT_TRUE(StringHelper::naturalLess("9", "10"));
    EXPECT_FALSE(StringHelper::naturalLess("10", "9"));
    EXPECT_TRUE(StringHelper::naturalLess("run 9", "Run 10"));
    EXPECT_TRUE(StringHelper::naturalLess("Aurora", "black hole"));
    EXPECT_TRUE(StringHelper::naturalLess("Prisma", "Prisma 2"));
    EXPECT_FALSE(StringHelper::naturalLess("Prisma 2", "Prisma"));
    EXPECT_TRUE(StringHelper::naturalLess("save_2", "save_10"));
    EXPECT_TRUE(StringHelper::naturalLess("7", "007") != StringHelper::naturalLess("007", "7"));
    EXPECT_FALSE(StringHelper::naturalLess("Aurora", "Aurora"));

    std::vector<std::string> names = {"15", "9", "Vortices", "10", "abyssal River", "Complexity"};
    std::ranges::sort(names, StringHelper::naturalLess);
    EXPECT_EQ((std::vector<std::string>{"9", "10", "15", "abyssal River", "Complexity", "Vortices"}), names);
}
