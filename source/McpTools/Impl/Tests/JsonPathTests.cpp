#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <Persister/Interface/SerializerService.h>

#include <McpTools/Impl/JsonPath.h>

class JsonPathTests : public ::testing::Test
{
protected:
    std::string getErrorMessage(std::string const& pathText, boost::json::value const& format)
    {
        try {
            JsonPath(pathText).resolveFormat(format);
        } catch (std::invalid_argument const& exception) {
            return exception.what();
        }
        return {};
    }

    boost::json::value getNodeFormat() const
    {
        return SerializerService::get().getJsonFormatOfGenome().at("genes").as_array().at(0).at("nodes").as_array().at(0);
    }
};

TEST_F(JsonPathTests, findField)
{
    auto json = boost::json::parse(R"({"cellType": {"sensor": {"maxRange": 80}}})");

    auto values = JsonPath("cellType.sensor.maxRange").find(json);

    ASSERT_EQ(1, values.size());
    EXPECT_EQ(80, values.front()->as_int64());
}

TEST_F(JsonPathTests, findAlternativeOfOtherType)
{
    auto json = boost::json::parse(R"({"cellType": {"muscle": {}}})");

    EXPECT_TRUE(JsonPath("cellType.sensor.maxRange").find(json).empty());
}

TEST_F(JsonPathTests, findArrayElements)
{
    auto json = boost::json::parse(R"({"biases": [1, 2, 3]})");

    EXPECT_EQ(3, JsonPath("biases[*]").find(json).size());
    ASSERT_EQ(1, JsonPath("biases[1]").find(json).size());
    EXPECT_EQ(2, JsonPath("biases[1]").find(json).front()->as_int64());
    EXPECT_TRUE(JsonPath("biases[3]").find(json).empty());
}

TEST_F(JsonPathTests, findAnyKey)
{
    auto json = boost::json::parse(R"({"cellType": {"sensor": {"mode": {"detectCreature": {"restrictToColors": 3}}}}})");

    auto values = JsonPath("cellType.*.mode.*.restrictToColors").find(json);

    ASSERT_EQ(1, values.size());
    EXPECT_EQ(3, values.front()->as_int64());
}

TEST_F(JsonPathTests, invalidPaths)
{
    EXPECT_THROW(JsonPath(""), std::invalid_argument);
    EXPECT_THROW(JsonPath("color."), std::invalid_argument);
    EXPECT_THROW(JsonPath("cellType..sensor"), std::invalid_argument);
    EXPECT_THROW(JsonPath("biases[x]"), std::invalid_argument);
    EXPECT_THROW(JsonPath("biases[-1]"), std::invalid_argument);
    EXPECT_THROW(JsonPath("biases[1"), std::invalid_argument);
}

TEST_F(JsonPathTests, resolveFormat)
{
    auto format = getNodeFormat();

    auto formats = JsonPath("cellType.sensor.maxRange").resolveFormat(format);

    ASSERT_EQ(1, formats.size());
    EXPECT_TRUE(formats.front()->is_number());
}

TEST_F(JsonPathTests, resolveFormat_unknownField)
{
    auto message = getErrorMessage("cellType.sensor.maxrange", getNodeFormat());

    EXPECT_NE(std::string::npos, message.find("'maxrange'"));
    EXPECT_NE(std::string::npos, message.find("maxRange"));
}

TEST_F(JsonPathTests, resolveFormat_unknownAlternative)
{
    auto message = getErrorMessage("cellType.teleporter", getNodeFormat());

    EXPECT_NE(std::string::npos, message.find("alternative"));
    EXPECT_NE(std::string::npos, message.find("sensor"));
}

TEST_F(JsonPathTests, conditions)
{
    auto json = boost::json::parse(R"({"color": 3, "values": [1.5, 2.5], "name": "worm", "limit": null})");
    auto isFulfilled = [&](std::string const& path, JsonConditionOperator op, boost::json::value const& value) {
        return JsonCondition{.path = JsonPath(path), .op = op, .value = value}.isFulfilled(json);
    };

    EXPECT_TRUE(isFulfilled("color", JsonConditionOperator::Equal, 3.0));
    EXPECT_FALSE(isFulfilled("color", JsonConditionOperator::NotEqual, 3));
    EXPECT_TRUE(isFulfilled("values[*]", JsonConditionOperator::Greater, 2));
    EXPECT_FALSE(isFulfilled("values[*]", JsonConditionOperator::Less, 1));
    EXPECT_TRUE(isFulfilled("name", JsonConditionOperator::In, boost::json::array{"plant", "worm"}));
    EXPECT_TRUE(isFulfilled("name", JsonConditionOperator::Exists, true));
    EXPECT_TRUE(isFulfilled("size", JsonConditionOperator::Exists, false));
    EXPECT_FALSE(isFulfilled("size", JsonConditionOperator::NotEqual, 1));
    EXPECT_FALSE(isFulfilled("limit", JsonConditionOperator::Exists, true));
    EXPECT_FALSE(isFulfilled("limit", JsonConditionOperator::NotEqual, 1));
    EXPECT_TRUE(isFulfilled("limit", JsonConditionOperator::Equal, nullptr));
}
