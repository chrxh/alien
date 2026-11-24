#include "McpToolsFacadeImpl.h"

#include <algorithm>
#include <format>

#include <boost/json.hpp>

#include <Base/LoggingService.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include "McpCreatorTools.h"
#include "McpDocumentationTools.h"
#include "McpEditTools.h"
#include "McpGenomeEditTools.h"
#include "McpGenomeTools.h"
#include "McpInspectionTools.h"
#include "McpMassOperationTools.h"
#include "McpMultiplierTools.h"
#include "McpNetworkTools.h"
#include "McpParameterTools.h"
#include "McpSelectionTools.h"
#include "McpSimulationTools.h"
#include "McpTemporalTools.h"
#include "McpViewTools.h"

namespace
{
    auto constexpr Instructions =
        "ALIEN is an artificial life simulation of soft bodies and fluids in which creatures grow from genomes, sense, move, feed and evolve. Before "
        "designing creatures, building worlds or tuning simulation parameters, read the relevant chapters of the documentation with read_documentation, "
        "e.g. 'how-life-works.md', 'cell-types.md', 'genomes.md', 'energy.md' and 'simulation-parameters.md'.";

    auto constexpr ServerName = "alien";
    auto constexpr PersisterPollInterval = std::chrono::milliseconds(10);
    auto constexpr MaxCommandLogEntries = 1000;
    auto constexpr MaxLogTextLength = 1000;
    auto constexpr MaxLogTextLines = 20;
}

void _McpToolsFacadeImpl::set(McpToolsFacade const& instance)
{
    _instance = instance;
}

_McpToolsFacadeImpl::_McpToolsFacadeImpl()
    : _executor([this](std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result) {
        addCommandLogEntry(toolName, arguments, result);
    })
{}

_McpToolsFacadeImpl::~_McpToolsFacadeImpl()
{
    _executor.beginShutdown();
    _server.reset();
}

void _McpToolsFacadeImpl::setup(McpToolContext& context)
{
    _context = &context;
    _tools = createTools(context);

    _toolGroups.clear();
    for (auto const& tool : _tools) {
        if (_toolGroups.empty() || _toolGroups.back().name != tool.group) {
            _toolGroups.emplace_back(McpToolGroup{.name = tool.group});
        }
        _toolGroups.back().toolNames.emplace_back(tool.name);
    }
}

void _McpToolsFacadeImpl::shutdown()
{
    stopServer();
}

bool _McpToolsFacadeImpl::startServer(int port)
{
    if (_server) {
        return true;
    }
    auto server = std::make_unique<McpServer>(ServerName, Const::ProgramVersion, Instructions, _executor.createToolsExecutedInProcess(_tools));
    if (!server->start(port)) {
        log(Priority::Important, std::format("mcp: could not start server on port {}", port));
        return false;
    }
    _server = std::move(server);

    log(Priority::Important, "mcp: server started at " + McpServer::createUrl(_server->getPort()));
    return true;
}

void _McpToolsFacadeImpl::stopServer()
{
    if (!_server) {
        return;
    }
    _executor.beginShutdown();
    _server.reset();
    _executor.endShutdown();

    log(Priority::Important, "mcp: server stopped");
}

bool _McpToolsFacadeImpl::isServerRunning() const
{
    return _server != nullptr;
}

void _McpToolsFacadeImpl::processToolCalls()
{
    McpTemporalTools::get().process();
    McpSimulationTools::get().process();
    McpNetworkTools::get().process();

    _executor.process();
}

void _McpToolsFacadeImpl::waitForToolCalls(std::chrono::milliseconds const& maxDuration)
{
    if (McpTemporalTools::get().isCalculatingTimesteps()) {
        return;
    }
    auto waitingForPersister = McpSimulationTools::get().hasPendingRequests() || McpNetworkTools::get().hasPendingRequests();
    _executor.waitForToolCalls(waitingForPersister ? std::min(maxDuration, PersisterPollInterval) : maxDuration);
}

std::vector<McpToolGroup> const& _McpToolsFacadeImpl::getToolGroups() const
{
    return _toolGroups;
}

std::deque<McpCommandLogEntry> const& _McpToolsFacadeImpl::getCommandLog() const
{
    return _commandLog;
}

void _McpToolsFacadeImpl::clearCommandLog()
{
    _commandLog.clear();
}

std::vector<McpTool> _McpToolsFacadeImpl::createTools(McpToolContext& context) const
{
    std::vector<McpTool> result;
    auto addGroup = [&result](std::string const& group, std::initializer_list<std::vector<McpTool>> toolLists) {
        for (auto const& tools : toolLists) {
            for (auto tool : tools) {
                tool.group = group;
                result.emplace_back(std::move(tool));
            }
        }
    };
    addGroup("Temporal control", {McpTemporalTools::get().getTools(context)});
    addGroup("Simulations and files", {McpSimulationTools::get().getTools(context), McpNetworkTools::get().getTools(context)});
    addGroup(
        "World editing",
        {McpCreatorTools::get().getTools(context),
         McpSelectionTools::get().getTools(context),
         McpEditTools::get().getTools(context),
         McpMultiplierTools::get().getTools(context),
         McpMassOperationTools::get().getTools(context),
         McpGenomeTools::get().getTools(context),
         McpGenomeEditTools::get().getTools(context)});
    addGroup("Inspection and view", {McpViewTools::get().getTools(context), McpInspectionTools::get().getTools(context)});
    addGroup("Simulation parameters", {McpParameterTools::get().getTools()});
    addGroup("Documentation", {McpDocumentationTools::get().getTools()});
    return result;
}

void _McpToolsFacadeImpl::addCommandLogEntry(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result)
{
    log(Priority::Important, result.isError ? std::format("mcp: {} -> error: {}", toolName, result.text) : "mcp: " + toolName);

    McpCommandLogEntry entry{
        .time = std::chrono::system_clock::now(),
        .toolName = toolName,
        .arguments = arguments.empty() ? std::string() : StringHelper::truncate(boost::json::serialize(arguments), MaxLogTextLength, MaxLogTextLines),
        .result = StringHelper::truncate(result.text, MaxLogTextLength, MaxLogTextLines),
        .isError = result.isError};
    _commandLog.emplace_back(entry);
    if (toInt(_commandLog.size()) > MaxCommandLogEntries) {
        _commandLog.pop_front();
    }
    _context->onCommandExecuted(entry);
}
