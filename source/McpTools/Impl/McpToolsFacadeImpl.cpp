#include "McpToolsFacadeImpl.h"

#include "McpCreatorTools.h"
#include "McpDocumentationTools.h"
#include "McpEditTools.h"
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
}

void _McpToolsFacadeImpl::set(McpToolsFacade const& instance)
{
    _instance = instance;
}

std::string _McpToolsFacadeImpl::getInstructions()
{
    return Instructions;
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
         McpMassOperationTools::get().getTools(context),
         McpGenomeTools::get().getTools(context)});
    addGroup("Inspection and view", {McpViewTools::get().getTools(context), McpInspectionTools::get().getTools(context)});
    addGroup("Simulation parameters", {McpParameterTools::get().getTools()});
    addGroup("Documentation", {McpDocumentationTools::get().getTools()});
    return result;
}

void _McpToolsFacadeImpl::process()
{
    McpTemporalTools::get().process();
    McpSimulationTools::get().process();
    McpNetworkTools::get().process();
}
