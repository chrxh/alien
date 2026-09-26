#pragma once

#include <functional>
#include <optional>

#include <Network/McpServer.h>

#include <PersisterInterface/Definitions.h>
#include <PersisterInterface/PersisterRequestId.h>
#include <PersisterInterface/SenderId.h>

class McpPersisterTask
{
public:
    void execute(
        std::function<PersisterRequestId(SenderId const&)> const& requestFunc,
        std::function<McpToolResult(PersisterRequestId const&)> const& finishFunc,
        McpToolCompletion const& completion);

    void process();

private:
    void complete(McpToolResult const& result);

    TaskProcessor _processor;
    std::optional<McpToolCompletion> _completion;
};
