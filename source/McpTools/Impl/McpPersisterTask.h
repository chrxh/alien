#pragma once

#include <functional>
#include <optional>

#include <Network/Interface/McpServer.h>

#include <Persister/Interface/Definitions.h>
#include <Persister/Interface/PersisterRequestId.h>
#include <Persister/Interface/SenderId.h>

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
