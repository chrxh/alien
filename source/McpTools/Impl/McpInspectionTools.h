#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Data/Descs.h>

#include <Network/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpInspectionTools
{
    MAKE_SINGLETON(McpInspectionTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

    boost::json::object createInspectionResult(ContentDesc const& inspectedData, std::vector<uint64_t> const& ids, bool includeGenomes, bool omitDefaultValues)
        const;

private:
    McpToolResult findObjects(boost::json::object const& arguments) const;
    McpToolResult inspectObjects(boost::json::object const& arguments) const;
    McpToolResult changeObject(boost::json::object const& arguments) const;
    McpToolResult getJsonFormat(boost::json::object const& arguments) const;

    boost::json::object describeBriefly(ObjectDesc const& object, float distance) const;
    boost::json::object describeBriefly(EnergyDesc const& energy, float distance) const;
    ExtendedObjectOrEnergyDesc getInspectedEntity(uint64_t id) const;

    McpToolContext* _context = nullptr;
};
