#include <stdexcept>

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <Network/McpArguments.h>

class McpArgumentsTests : public ::testing::Test
{
protected:
    boost::json::object parse(std::string const& json) { return boost::json::parse(json).as_object(); }
};

TEST_F(McpArgumentsTests, float_fromInteger)
{
    EXPECT_EQ(3.0f, McpArguments::getFloat(parse(R"({"x": 3})"), "x"));
}

TEST_F(McpArgumentsTests, float_outOfRange)
{
    EXPECT_THROW(McpArguments::getFloat(parse(R"({"x": 1.5})"), "x", 0.0f, 1.0f), std::invalid_argument);
}

TEST_F(McpArgumentsTests, float_missing)
{
    EXPECT_THROW(McpArguments::getFloat(parse("{}"), "x"), std::invalid_argument);
    EXPECT_FALSE(McpArguments::getOptionalFloat(parse("{}"), "x").has_value());
}

TEST_F(McpArgumentsTests, int_fromWholeDouble)
{
    EXPECT_EQ(1000, McpArguments::getInt(parse(R"({"n": 1000.0})"), "n"));
}

TEST_F(McpArgumentsTests, int_fraction)
{
    EXPECT_THROW(McpArguments::getInt(parse(R"({"n": 2.5})"), "n"), std::invalid_argument);
}

TEST_F(McpArgumentsTests, id_fromString)
{
    EXPECT_EQ(18446744073709551615ull, McpArguments::getId(parse(R"({"id": "18446744073709551615"})"), "id"));
}

TEST_F(McpArgumentsTests, id_fromInteger)
{
    EXPECT_EQ(42, McpArguments::getId(parse(R"({"id": 42})"), "id"));
}

TEST_F(McpArgumentsTests, id_invalid)
{
    EXPECT_THROW(McpArguments::getId(parse(R"({"id": "12a"})"), "id"), std::invalid_argument);
    EXPECT_THROW(McpArguments::getId(parse(R"({"id": ""})"), "id"), std::invalid_argument);
    EXPECT_THROW(McpArguments::getId(parse(R"({"id": -1})"), "id"), std::invalid_argument);
}

TEST_F(McpArgumentsTests, ids)
{
    EXPECT_EQ((std::vector<uint64_t>{1, 2}), McpArguments::getIds(parse(R"({"ids": ["1", 2]})"), "ids", 1));
    EXPECT_THROW(McpArguments::getIds(parse(R"({"ids": []})"), "ids", 1), std::invalid_argument);
}

TEST_F(McpArgumentsTests, ints)
{
    EXPECT_EQ((std::vector<int>{3, 1}), McpArguments::getInts(parse(R"({"n": [3, 1]})"), "n", 1, 0, 6));
    EXPECT_THROW(McpArguments::getInts(parse(R"({"n": []})"), "n", 1), std::invalid_argument);
    EXPECT_THROW(McpArguments::getInts(parse(R"({"n": [7]})"), "n", 1, 0, 6), std::invalid_argument);
    EXPECT_THROW(McpArguments::getInts(parse(R"({"n": 3})"), "n", 1), std::invalid_argument);
}

TEST_F(McpArgumentsTests, int_wrongType)
{
    EXPECT_THROW(McpArguments::getInt(parse(R"({"n": "ten"})"), "n"), std::invalid_argument);
}

TEST_F(McpArgumentsTests, bool_wrongType)
{
    EXPECT_THROW(McpArguments::getBool(parse(R"({"b": 1})"), "b"), std::invalid_argument);
}

TEST_F(McpArgumentsTests, filePath_nonAscii)
{
    auto path = McpArguments::getFilePath(parse(R"({"f": "C:/J\u00fcrgen/x.settings.json"})"), "f");

    EXPECT_TRUE(path.u8string() == u8"C:/J\u00fcrgen/x.settings.json");
}

TEST_F(McpArgumentsTests, points)
{
    auto points = McpArguments::getPoints(parse(R"({"p": [[1, 2], [3.5, 4]]})"), "p", 2);

    ASSERT_EQ(2, points.size());
    EXPECT_EQ((RealVector2D{3.5f, 4.0f}), points.at(1));
}

TEST_F(McpArgumentsTests, points_tooFew)
{
    EXPECT_THROW(McpArguments::getPoints(parse(R"({"p": [[1, 2]]})"), "p", 2), std::invalid_argument);
}

TEST_F(McpArgumentsTests, points_invalidPair)
{
    EXPECT_THROW(McpArguments::getPoints(parse(R"({"p": [[1, 2, 3], [4, 5]]})"), "p", 1), std::invalid_argument);
}

TEST_F(McpArgumentsTests, objects)
{
    auto objects = McpArguments::getObjects(parse(R"({"o": [{"a": 1}, {"b": 2}]})"), "o", 1);

    ASSERT_EQ(2, objects.size());
    EXPECT_EQ(1, McpArguments::getInt(objects.at(0), "a"));
    EXPECT_EQ(2, McpArguments::getInt(objects.at(1), "b"));
}

TEST_F(McpArgumentsTests, objects_invalidEntry)
{
    EXPECT_THROW(McpArguments::getObjects(parse(R"({"o": [{"a": 1}, 2]})"), "o", 1), std::invalid_argument);
}

TEST_F(McpArgumentsTests, objects_tooFew)
{
    EXPECT_THROW(McpArguments::getObjects(parse(R"({"o": []})"), "o", 1), std::invalid_argument);
}
