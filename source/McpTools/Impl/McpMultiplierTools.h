#pragma once

#include <optional>
#include <vector>

#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>
#include <Data/Interface/MultiplierService.h>

#include <Engine/Interface/SelectionShallowData.h>

#include <Network/Interface/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpMultiplierTools
{
    MAKE_SINGLETON(McpMultiplierTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult multiplyInGrid(boost::json::object const& arguments);
    McpToolResult multiplyRandomly(boost::json::object const& arguments);
    McpToolResult undoMultiplication();

    void checkSelectionForMultiplication(float numCopies) const;
    void storeForUndo(ContentDesc&& origSelection);
    std::string describeSelection() const;

    McpToolContext* _context = nullptr;
    std::optional<ContentDesc> _origSelection;
    std::optional<SelectionShallowData> _selectionAfterMultiplication;
};
