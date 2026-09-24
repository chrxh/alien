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

TEST_F(McpArgumentsTests, int_wrongType)
{
    EXPECT_THROW(McpArguments::getInt(parse(R"({"n": "ten"})"), "n"), std::invalid_argument);
}

TEST_F(McpArgumentsTests, bool_wrongType)
{
    EXPECT_THROW(McpArguments::getBool(parse(R"({"b": 1})"), "b"), std::invalid_argument);
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
