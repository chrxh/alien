#pragma once

#include <optional>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/SelectionShallowData.h>

#include <Network/McpServer.h>

#include "MultiplierService.h"

class McpMultiplierTools
{
    MAKE_SINGLETON(McpMultiplierTools);

public:
    std::vector<McpTool> getTools();

private:
    McpToolResult multiplyInGrid(boost::json::object const& arguments);
    McpToolResult multiplyRandomly(boost::json::object const& arguments);
    McpToolResult undoMultiplication();

    void checkSelectionForMultiplication(int numCopies) const;
    void storeForUndo(MultiplierService::Result&& result);
    std::string describeSelection() const;

    std::optional<ContentDesc> _origSelection;
    std::optional<SelectionShallowData> _selectionAfterMultiplication;
};
