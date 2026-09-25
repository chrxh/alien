#include "McpServer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string_view>

#include <boost/json.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>

namespace
{
    auto constexpr Host = "127.0.0.1";
    auto constexpr Endpoint = "/mcp";
    auto constexpr JsonRpcVersion = "2.0";
    auto constexpr SupportedProtocolVersions = std::to_array<std::string_view>({"2025-11-25", "2025-06-18", "2025-03-26"});
    auto constexpr DefaultProtocolVersion = "2025-06-18";
    auto constexpr MaxStartupTime = std::chrono::milliseconds(2000);
    auto constexpr StartupPollInterval = std::chrono::milliseconds(1);
    auto constexpr KeepAliveTimeoutSec = 1;

    auto constexpr ParseError = -32700;
    auto constexpr InvalidRequest = -32600;
    auto constexpr MethodNotFound = -32601;
    auto constexpr InvalidParams = -32602;

    struct JsonRpcError
    {
        int code = 0;
        std::string message;
    };

    std::string createErrorResponse(boost::json::value const& id, int code, std::string const& message)
    {
        return boost::json::serialize(boost::json::object{
            {"jsonrpc", JsonRpcVersion},
            {"id", id},
            {"error", boost::json::object{{"code", code}, {"message", message}}},
        });
    }

    std::string createResultResponse(boost::json::value const& id, boost::json::object result)
    {
        return boost::json::serialize(boost::json::object{{"jsonrpc", JsonRpcVersion}, {"id", id}, {"result", std::move(result)}});
    }

    boost::json::object initialize(boost::json::object const& params, std::string const& serverName, std::string const& serverVersion)
    {
        std::string protocolVersion = DefaultProtocolVersion;
        if (auto requestedVersion = params.if_contains("protocolVersion"); requestedVersion && requestedVersion->is_string()) {
            auto const& version = requestedVersion->as_string();
            if (std::ranges::find(SupportedProtocolVersions, std::string_view(version)) != SupportedProtocolVersions.end()) {
                protocolVersion = version;
            }
        }
        return {
            {"protocolVersion", protocolVersion},
            {"capabilities", boost::json::object{{"tools", boost::json::object{{"listChanged", false}}}}},
            {"serverInfo", boost::json::object{{"name", serverName}, {"version", serverVersion}}},
        };
    }

    boost::json::object listTools(std::vector<McpTool> const& tools)
    {
        boost::json::array toolList;
        for (auto const& tool : tools) {
            toolList.emplace_back(boost::json::object{{"name", tool.name}, {"description", tool.description}, {"inputSchema", tool.inputSchema}});
        }
        return {{"tools", std::move(toolList)}};
    }

    boost::json::object callTool(boost::json::object const& params, std::vector<McpTool> const& tools)
    {
        auto name = params.if_contains("name");
        if (!name || !name->is_string()) {
            throw JsonRpcError{InvalidParams, "Missing tool name"};
        }
        auto tool = std::ranges::find_if(tools, [&](auto const& candidate) { return candidate.name == name->as_string(); });
        if (tool == tools.end()) {
            throw JsonRpcError{InvalidParams, "Unknown tool: " + std::string(name->as_string())};
        }
        auto arguments = params.if_contains("arguments");
        if (arguments && !arguments->is_object()) {
            throw JsonRpcError{InvalidParams, "Tool arguments must be an object"};
        }

        auto result = [&] {
            try {
                return tool->handler(arguments ? arguments->as_object() : boost::json::object());
            } catch (std::exception const& exception) {
                return McpToolResult{.text = exception.what(), .isError = true};
            }
        }();
        return {
            {"content", boost::json::array{boost::json::object{{"type", "text"}, {"text", result.text}}}},
            {"isError", result.isError},
        };
    }

    bool isOriginAllowed(httplib::Request const& request)
    {
        if (!request.has_header("Origin")) {
            return true;
        }
        auto origin = request.get_header_value("Origin");
        auto hostStart = origin.find("://");
        if (hostStart == std::string::npos) {
            return false;
        }
        auto host = origin.substr(hostStart + 3);
        if (host.starts_with("[::1]")) {
            return true;
        }
        host = host.substr(0, host.find_first_of(":/"));
        return host == "localhost" || host == "127.0.0.1";
    }
}

McpServer::McpServer(std::string const& serverName, std::string const& serverVersion, std::vector<McpTool> tools)
    : _serverName(serverName)
    , _serverVersion(serverVersion)
    , _tools(std::move(tools))
{}

McpServer::~McpServer()
{
    stop();
}

namespace
{
    void setSocketOptions(socket_t socket)
    {
#ifdef _WIN32
        httplib::default_socket_options(socket);
#else
        int yes = 1;
        setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<void*>(&yes), sizeof(yes));
#endif
    }
}

bool McpServer::start(int port)
{
    stop();

    auto server = std::make_unique<httplib::Server>();
    server->set_socket_options(setSocketOptions);

    server->set_keep_alive_max_count(1);
    server->set_keep_alive_timeout(KeepAliveTimeoutSec);

    server->Post(Endpoint, [this](httplib::Request const& request, httplib::Response& response) {
        if (!isOriginAllowed(request)) {
            response.status = 403;
            return;
        }
        if (auto message = handleMessage(request.body)) {
            response.set_content(*message, "application/json");
        } else {
            response.status = 202;
        }
    });
    auto methodNotAllowed = [](httplib::Request const&, httplib::Response& response) {
        response.status = 405;
        response.set_header("Allow", "POST");
    };
    server->Get(Endpoint, methodNotAllowed);
    server->Delete(Endpoint, methodNotAllowed);

    if (port == 0) {
        port = server->bind_to_any_port(Host);
        if (port <= 0) {
            return false;
        }
    } else if (!server->bind_to_port(Host, port)) {
        return false;
    }

    _port = port;
    _server = std::move(server);
    _thread = std::thread([server = _server.get()] { server->listen_after_bind(); });

    for (auto waitTime = std::chrono::milliseconds(0); !_server->is_running() && waitTime < MaxStartupTime; waitTime += StartupPollInterval) {
        std::this_thread::sleep_for(StartupPollInterval);
    }
    return true;
}

void McpServer::stop()
{
    if (!_server) {
        return;
    }
    _server->stop();
    _thread.join();
    _server.reset();
    _port = 0;
}

bool McpServer::isRunning() const
{
    return _server != nullptr;
}

int McpServer::getPort() const
{
    return _port;
}

std::optional<std::string> McpServer::handleMessage(std::string const& message) const
{
    boost::json::value parsedMessage;
    try {
        parsedMessage = boost::json::parse(message);
    } catch (...) {
        return createErrorResponse(nullptr, ParseError, "Parse error");
    }
    if (!parsedMessage.is_object()) {
        return createErrorResponse(nullptr, InvalidRequest, "Expected a single JSON-RPC message");
    }

    auto const& request = parsedMessage.as_object();
    auto method = request.if_contains("method");
    auto id = request.if_contains("id");
    if (!method || !id) {
        return std::nullopt;
    }
    if (!method->is_string()) {
        return createErrorResponse(*id, InvalidRequest, "Method must be a string");
    }
    auto params = request.if_contains("params");
    auto const& paramsObject = params && params->is_object() ? params->as_object() : boost::json::object();

    try {
        auto const& methodName = method->as_string();
        if (methodName == "initialize") {
            return createResultResponse(*id, initialize(paramsObject, _serverName, _serverVersion));
        }
        if (methodName == "ping") {
            return createResultResponse(*id, {});
        }
        if (methodName == "tools/list") {
            return createResultResponse(*id, listTools(_tools));
        }
        if (methodName == "tools/call") {
            return createResultResponse(*id, callTool(paramsObject, _tools));
        }
        throw JsonRpcError{MethodNotFound, "Method not found: " + std::string(methodName)};
    } catch (JsonRpcError const& error) {
        return createErrorResponse(*id, error.code, error.message);
    }
}
