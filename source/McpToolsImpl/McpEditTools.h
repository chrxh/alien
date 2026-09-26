#pragma once

#include <optional>
#include <vector>

#include <Base/Singleton.h>

#include <Data/Descs.h>

#include <EngineInterface/SelectionShallowData.h>

#include <Network/McpServer.h>
#include <McpToolsInterface/McpToolContext.h>

class McpEditTools
{
    MAKE_SINGLETON(McpEditTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult selectAt(boost::json::object const& arguments) const;
    McpToolResult glueSelection(boost::json::object const& arguments) const;
    McpToolResult connectSelectionToSurroundings() const;
    McpToolResult cutConnections(boost::json::object const& arguments) const;
    McpToolResult setSelectionVelocity(boost::json::object const& arguments) const;
    McpToolResult setSelectionAngularVelocity(boost::json::object const& arguments) const;
    McpToolResult uniformSelectionVelocities(boost::json::object const& arguments) const;
    McpToolResult copySelection(boost::json::object const& arguments);
    McpToolResult pasteSelection(boost::json::object const& arguments) const;
    McpToolResult applyForce(boost::json::object const& arguments) const;

    SelectionShallowData getNonEmptySelection() const;

    McpToolContext* _context = nullptr;
    std::optional<ContentDesc> _copiedSelection;
};
