#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Data/SimulationParametersTypes.h>

#include <Network/McpServer.h>

class McpParameterTools
{
    MAKE_SINGLETON(McpParameterTools);

public:
    std::vector<McpTool> getTools();

private:
    McpToolResult listParameterGroups(boost::json::object const& arguments) const;
    McpToolResult getParameters(boost::json::object const& arguments) const;
    McpToolResult setParameters(boost::json::object const& arguments) const;
    McpToolResult enableExpertSettings(boost::json::object const& arguments) const;
    McpToolResult resetParameters(boost::json::object const& arguments) const;

    McpToolResult listLocations() const;
    McpToolResult addLocation(boost::json::object const& arguments, LocationType locationType) const;
    McpToolResult cloneLocation(boost::json::object const& arguments) const;
    McpToolResult deleteLocation(boost::json::object const& arguments) const;
    McpToolResult moveLocation(boost::json::object const& arguments) const;

    McpToolResult loadParameters(boost::json::object const& arguments) const;
    McpToolResult saveParameters(boost::json::object const& arguments) const;
};
