#pragma once

#include <McpToolsInterface/McpToolsFacade.h>

class _McpToolsFacadeImpl : public _McpToolsFacade
{
public:
    static void set(McpToolsFacade const& instance);

    std::vector<McpTool> getTools(McpToolContext& context) override;
};
