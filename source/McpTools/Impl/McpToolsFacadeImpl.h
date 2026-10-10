#pragma once

#include <McpTools/Interface/McpToolsFacade.h>

class _McpToolsFacadeImpl : public _McpToolsFacade
{
public:
    static void set(McpToolsFacade const& instance);

    std::vector<McpTool> getTools(McpToolContext& context) override;
    void process() override;
};
