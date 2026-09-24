#pragma once

#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/SelectionShallowData.h>

#include "McpHost.h"
#include "McpServer.h"

class McpSelectionTools
{
    MAKE_SINGLETON(McpSelectionTools);

public:
    std::vector<McpTool> getTools(McpHost const& host);

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

    SelectionShallowData getNonEmptySelection() const;
    std::string describeSelection(SelectionShallowData const& selection) const;

    McpHost _host;
};
