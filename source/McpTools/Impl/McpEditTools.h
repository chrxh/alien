#pragma once

#include <optional>
#include <vector>

#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>

#include <Engine/Interface/SelectionShallowData.h>

#include <Network/Interface/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

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
