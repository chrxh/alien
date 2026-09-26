#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Data/Descs.h>

#include <Network/McpServer.h>
#include <McpToolsInterface/McpToolContext.h>

class McpInspectionTools
{
    MAKE_SINGLETON(McpInspectionTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult findObjects(boost::json::object const& arguments) const;
    McpToolResult inspectObjects(boost::json::object const& arguments) const;
    McpToolResult changeObject(boost::json::object const& arguments) const;
    McpToolResult getJsonFormat(boost::json::object const& arguments) const;

    boost::json::object describeBriefly(ExtendedObjectOrEnergyDesc const& entity, RealVector2D const& center) const;
    ExtendedObjectOrEnergyDesc getInspectedEntity(uint64_t id) const;

    McpToolContext* _context = nullptr;
};
