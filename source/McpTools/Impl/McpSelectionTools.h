#pragma once

#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <Engine/Interface/SelectionShallowData.h>

#include <Network/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpSelectionTools
{
    MAKE_SINGLETON(McpSelectionTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult selectArea(boost::json::object const& arguments) const;
    McpToolResult clearSelection() const;
    McpToolResult getSelection() const;
    McpToolResult deleteSelection(boost::json::object const& arguments) const;
    McpToolResult fixSelection(boost::json::object const& arguments) const;
    McpToolResult colorSelection(boost::json::object const& arguments) const;
    McpToolResult setSelectionSticky(boost::json::object const& arguments) const;
    McpToolResult moveSelection(boost::json::object const& arguments) const;
    McpToolResult rotateSelection(boost::json::object const& arguments) const;
    McpToolResult relaxSelection(boost::json::object const& arguments) const;

    // Throws if nothing is selected, so tools also call it just as a guard
    SelectionShallowData getNonEmptySelection() const;
    std::string describeSelection(SelectionShallowData const& selection) const;

    McpToolContext* _context = nullptr;
};
