#include "McpToolsFacadeImpl.h"

#include "McpCreatorTools.h"
#include "McpEditTools.h"
#include "McpGenomeTools.h"
#include "McpInspectionTools.h"
#include "McpMultiplierTools.h"
#include "McpNetworkTools.h"
#include "McpParameterTools.h"
#include "McpSelectionTools.h"
#include "McpSimulationTools.h"
#include "McpTemporalTools.h"
#include "McpViewTools.h"

void _McpToolsFacadeImpl::set(McpToolsFacade const& instance)
{
    _instance = instance;
}

std::vector<McpTool> _McpToolsFacadeImpl::getTools(McpToolContext& context)
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
         McpGenomeTools::get().getTools(context)});
    addGroup("Inspection and view", {McpViewTools::get().getTools(context), McpInspectionTools::get().getTools(context)});
    addGroup("Simulation parameters", {McpParameterTools::get().getTools()});
    return result;
}

void _McpToolsFacadeImpl::process()
{
    McpTemporalTools::get().process();
    McpSimulationTools::get().process();
    McpNetworkTools::get().process();
}
