#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <Network/McpJson.h>

TEST(McpJsonTests, numbers)
{
    auto json = boost::json::array{0.5, 150.0, 0.1f, 1e-5, -3, "text"};

    EXPECT_EQ(R"([0.5,150,0.1,1e-05,-3,"text"])", McpJson::serialize(json));
}

TEST(McpJsonTests, nested)
{
    auto json = boost::json::object{{"a", boost::json::array{1.25, true, nullptr}}, {"b\"", boost::json::object{{"c", 2.0}}}};

    auto serialized = McpJson::serialize(json);

    EXPECT_EQ(R"({"a":[1.25,true,null],"b\"":{"c":2}})", serialized);
    EXPECT_NO_THROW(boost::json::parse(serialized));
}
