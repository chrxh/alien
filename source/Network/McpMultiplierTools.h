#pragma once

#include <optional>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/MultiplierService.h>
#include <EngineInterface/SelectionShallowData.h>

#include "McpHost.h"
#include "McpServer.h"

class McpMultiplierTools
{
    MAKE_SINGLETON(McpMultiplierTools);

public:
    std::vector<McpTool> getTools(McpHost const& host);

private:
    McpToolResult multiplyInGrid(boost::json::object const& arguments);
    McpToolResult multiplyRandomly(boost::json::object const& arguments);
    McpToolResult undoMultiplication();

    void checkSelectionForMultiplication(int numCopies) const;
    void storeForUndo(MultiplierService::Result&& result);
    std::string describeSelection() const;

    McpHost _host;
    std::optional<ContentDesc> _origSelection;
    std::optional<SelectionShallowData> _selectionAfterMultiplication;
};
