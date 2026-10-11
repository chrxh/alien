#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <boost/json/object.hpp>

namespace httplib
{
    class Server;
}

struct McpImage
{
    std::string mimeType;
    std::string data;
};

struct McpToolResult
{
    std::string text;
    std::vector<McpImage> images;
    bool isError = false;
};

using McpToolCompletion = std::function<void(McpToolResult const& result)>;

struct McpTool
{
    std::string name;
    std::string group;
    std::string description;
    boost::json::object inputSchema;
    std::function<McpToolResult(boost::json::object const& arguments)> handler;
    std::function<void(boost::json::object const& arguments, McpToolCompletion const& completion)> deferredHandler;
};

class McpServer
{
public:
    McpServer(std::string const& serverName, std::string const& serverVersion, std::vector<McpTool> tools);
    ~McpServer();

    bool start(int port);
    void stop();
    bool isRunning() const;
    int getPort() const;

    std::optional<std::string> handleMessage(std::string const& message) const;

private:
    std::string _serverName;
    std::string _serverVersion;
    std::vector<McpTool> _tools;

    std::unique_ptr<httplib::Server> _server;
    std::thread _thread;
    int _port = 0;
};
