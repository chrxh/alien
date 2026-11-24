#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <future>

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <McpTools/Impl/McpToolExecutor.h>

class McpToolExecutorTests : public ::testing::Test
{
public:
    McpToolExecutorTests()
        : _executor([this](std::string const& toolName, boost::json::object const&, McpToolResult const& result) {
            _reportedResults.emplace_back(toolName + ": " + result.text);
        })
    {
        _tools = _executor.createToolsExecutedInProcess({
            McpTool{
                .name = "echo",
                .handler =
                    [this](boost::json::object const& arguments) {
                        _executingThreadId = std::this_thread::get_id();
                        return McpToolResult{.text = std::string(arguments.at("text").as_string())};
                    },
            },
            McpTool{
                .name = "fail",
                .handler = [](boost::json::object const&) -> McpToolResult { throw std::runtime_error("Failed"); },
            },
            McpTool{
                .name = "defer",
                .deferredHandler = [this](boost::json::object const&, McpToolCompletion const& completion) { _completion = completion; },
            },
        });
    }

protected:
    std::future<McpToolResult> callTool(std::string const& name, boost::json::object const& arguments = {})
    {
        auto tool = std::ranges::find(_tools, name, &McpTool::name);
        return std::async(std::launch::async, [handler = tool->handler, arguments] { return handler(arguments); });
    }

    void waitAndProcess()
    {
        _executor.waitForToolCalls(MaxWaitDuration);
        _executor.process();
    }

    static auto constexpr MaxWaitDuration = std::chrono::milliseconds(10000);

    std::vector<std::string> _reportedResults;
    std::thread::id _executingThreadId;
    McpToolCompletion _completion;
    McpToolExecutor _executor;
    std::vector<McpTool> _tools;
};

TEST_F(McpToolExecutorTests, toolIsExecutedOnProcessingThread)
{
    auto result = callTool("echo", {{"text", "hello"}});
    waitAndProcess();

    EXPECT_EQ("hello", result.get().text);
    EXPECT_EQ(std::this_thread::get_id(), _executingThreadId);
    EXPECT_EQ(std::vector<std::string>{"echo: hello"}, _reportedResults);
}

TEST_F(McpToolExecutorTests, toolIsNotExecutedBeforeProcessing)
{
    auto result = callTool("echo", {{"text", "hello"}});
    _executor.waitForToolCalls(MaxWaitDuration);

    EXPECT_EQ(std::future_status::timeout, result.wait_for(std::chrono::milliseconds(100)));
    EXPECT_TRUE(_reportedResults.empty());

    _executor.process();
    EXPECT_EQ("hello", result.get().text);
}

TEST_F(McpToolExecutorTests, exceptionBecomesErrorResult)
{
    auto futureResult = callTool("fail");
    waitAndProcess();

    auto result = futureResult.get();
    EXPECT_TRUE(result.isError);
    EXPECT_EQ("Failed", result.text);
    EXPECT_EQ(std::vector<std::string>{"fail: Failed"}, _reportedResults);
}

TEST_F(McpToolExecutorTests, deferredToolReturnsAfterCompletion)
{
    auto result = callTool("defer");
    waitAndProcess();

    EXPECT_EQ(std::future_status::timeout, result.wait_for(std::chrono::milliseconds(100)));
    EXPECT_TRUE(_reportedResults.empty());

    _completion(McpToolResult{.text = "completed"});
    EXPECT_EQ("completed", result.get().text);
    EXPECT_EQ(std::vector<std::string>{"defer: completed"}, _reportedResults);
}

TEST_F(McpToolExecutorTests, waitingEndsWithoutToolCalls)
{
    auto maxDuration = std::chrono::milliseconds(50);
    auto startTimepoint = std::chrono::steady_clock::now();
    _executor.waitForToolCalls(maxDuration);

    EXPECT_GE(std::chrono::steady_clock::now() - startTimepoint, maxDuration);
}

TEST_F(McpToolExecutorTests, shutdownAbortsWaitingToolCalls)
{
    auto result = callTool("echo", {{"text", "hello"}});
    _executor.waitForToolCalls(MaxWaitDuration);
    _executor.beginShutdown();

    EXPECT_TRUE(result.get().isError);
    EXPECT_TRUE(callTool("echo", {{"text", "hello"}}).get().isError);

    _executor.endShutdown();
    _executor.process();
    EXPECT_TRUE(_reportedResults.empty());
}

TEST_F(McpToolExecutorTests, toolCallsAreExecutedAfterShutdown)
{
    _executor.beginShutdown();
    _executor.endShutdown();

    auto result = callTool("echo", {{"text", "hello"}});
    waitAndProcess();

    EXPECT_EQ("hello", result.get().text);
}
