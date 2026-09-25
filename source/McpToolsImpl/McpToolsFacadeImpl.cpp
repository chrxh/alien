#include "McpToolsFacadeImpl.h"

#include "McpCreatorTools.h"
#include "McpMultiplierTools.h"
#include "McpParameterTools.h"
#include "McpSelectionTools.h"
#include "McpSimulationTools.h"

void _McpToolsFacadeImpl::set(McpToolsFacade const& instance)
{
    _instance = instance;
}

std::vector<McpTool> _McpToolsFacadeImpl::getTools(McpToolContext& context)
{
    std::vector<McpTool> result;
    for (auto const& tools :
         {McpSimulationTools::get().getTools(context),
          McpCreatorTools::get().getTools(context),
          McpSelectionTools::get().getTools(context),
          McpMultiplierTools::get().getTools(context),
          McpParameterTools::get().getTools()}) {
        result.insert(result.end(), tools.begin(), tools.end());
    }
    return result;
}
