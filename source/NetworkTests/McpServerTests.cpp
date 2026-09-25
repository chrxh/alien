#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <Network/McpServer.h>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>

class McpServerTests : public ::testing::Test
{
public:
    McpServerTests()
        : _server(
              "alien",
              "1.0",
              {McpTool{
                  .name = "echo",
                  .description = "Returns the text argument",
                  .inputSchema = {{"type", "object"}},
                  .handler =
                      [](boost::json::object const& arguments) {
                          if (!arguments.contains("text")) {
                              throw std::runtime_error("Missing text");
                          }
                          return McpToolResult{.text = std::string(arguments.at("text").as_string())};
                      },
              }})
    {}

protected:
    boost::json::object sendRequest(std::string const& method, boost::json::object const& params = {})
    {
        auto response =
            _server.handleMessage(boost::json::serialize(boost::json::object{{"jsonrpc", "2.0"}, {"id", 1}, {"method", method}, {"params", params}}));
        EXPECT_TRUE(response.has_value());
        return boost::json::parse(*response).as_object();
    }

    McpServer _server;
};

TEST_F(McpServerTests, initialize_supportedVersion)
{
    auto response = sendRequest("initialize", {{"protocolVersion", "2025-03-26"}});

    auto const& result = response.at("result").as_object();
    EXPECT_EQ("2025-03-26", result.at("protocolVersion").as_string());
    EXPECT_EQ("alien", result.at("serverInfo").as_object().at("name").as_string());
    EXPECT_TRUE(result.at("capabilities").as_object().contains("tools"));
}

TEST_F(McpServerTests, initialize_unsupportedVersion)
{
    auto response = sendRequest("initialize", {{"protocolVersion", "1999-01-01"}});

    EXPECT_EQ("2025-06-18", response.at("result").as_object().at("protocolVersion").as_string());
}

TEST_F(McpServerTests, notification_noResponse)
{
    auto response = _server.handleMessage(R"({"jsonrpc": "2.0", "method": "notifications/initialized"})");

    EXPECT_FALSE(response.has_value());
}

TEST_F(McpServerTests, listTools)
{
    auto response = sendRequest("tools/list");

    auto const& tools = response.at("result").as_object().at("tools").as_array();
    ASSERT_EQ(1, tools.size());
    EXPECT_EQ("echo", tools.at(0).as_object().at("name").as_string());
    EXPECT_TRUE(tools.at(0).as_object().contains("inputSchema"));
}

TEST_F(McpServerTests, callTool)
{
    auto response = sendRequest("tools/call", {{"name", "echo"}, {"arguments", boost::json::object{{"text", "hello"}}}});

    auto const& result = response.at("result").as_object();
    EXPECT_FALSE(result.at("isError").as_bool());
    EXPECT_EQ("hello", result.at("content").as_array().at(0).as_object().at("text").as_string());
}

TEST_F(McpServerTests, callTool_failingHandler)
{
    auto response = sendRequest("tools/call", {{"name", "echo"}});

    auto const& result = response.at("result").as_object();
    EXPECT_TRUE(result.at("isError").as_bool());
    EXPECT_EQ("Missing text", result.at("content").as_array().at(0).as_object().at("text").as_string());
}

TEST_F(McpServerTests, callTool_unknownTool)
{
    auto response = sendRequest("tools/call", {{"name", "unknown"}});

    EXPECT_EQ(-32602, response.at("error").as_object().at("code").as_int64());
}

TEST_F(McpServerTests, unknownMethod)
{
    auto response = sendRequest("resources/list");

    EXPECT_EQ(-32601, response.at("error").as_object().at("code").as_int64());
}

TEST_F(McpServerTests, invalidJson)
{
    auto response = _server.handleMessage("{invalid");

    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(-32700, boost::json::parse(*response).as_object().at("error").as_object().at("code").as_int64());
}

TEST_F(McpServerTests, http)
{
    ASSERT_TRUE(_server.start(0));
    httplib::Client client("127.0.0.1", _server.getPort());

    auto requestResult = client.Post("/mcp", R"({"jsonrpc": "2.0", "id": 7, "method": "ping"})", "application/json");
    ASSERT_TRUE(requestResult);
    EXPECT_EQ(200, requestResult->status);
    EXPECT_EQ(7, boost::json::parse(requestResult->body).as_object().at("id").as_int64());

    auto notificationResult = client.Post("/mcp", R"({"jsonrpc": "2.0", "method": "notifications/initialized"})", "application/json");
    ASSERT_TRUE(notificationResult);
    EXPECT_EQ(202, notificationResult->status);

    auto getResult = client.Get("/mcp");
    ASSERT_TRUE(getResult);
    EXPECT_EQ(405, getResult->status);

    auto foreignOriginResult = client.Post("/mcp", {{"Origin", "http://example.com"}}, R"({"jsonrpc": "2.0", "id": 1, "method": "ping"})", "application/json");
    ASSERT_TRUE(foreignOriginResult);
    EXPECT_EQ(403, foreignOriginResult->status);

    _server.stop();
    EXPECT_FALSE(_server.isRunning());
}
