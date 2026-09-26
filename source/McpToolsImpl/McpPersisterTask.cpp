#include "McpPersisterTask.h"

#include <stdexcept>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/TaskProcessor.h>

void McpPersisterTask::execute(
    std::function<PersisterRequestId(SenderId const&)> const& requestFunc,
    std::function<McpToolResult(PersisterRequestId const&)> const& finishFunc,
    McpToolCompletion const& completion)
{
    if (_completion) {
        throw std::runtime_error("The previous request of this kind is still in progress.");
    }
    if (!_processor) {
        _processor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    }
    _completion = completion;
    try {
        _processor->executeTask(
            requestFunc,
            [this, finishFunc](PersisterRequestId const& requestId) {
                try {
                    complete(finishFunc(requestId));
                } catch (std::exception const& exception) {
                    complete({.text = exception.what(), .isError = true});
                }
            },
            [this](std::vector<PersisterErrorInfo> const& errors) {
                std::string message;
                for (auto const& error : errors) {
                    message += (message.empty() ? "" : "\n") + error.message;
                }
                complete({.text = message, .isError = true});
            });
    } catch (...) {
        _completion.reset();
        throw;
    }
}

void McpPersisterTask::process()
{
    if (!_processor) {
        return;
    }
    _processor->process();
    if (_completion && !_processor->pendingTasks()) {
        complete({.text = "The request was aborted.", .isError = true});
    }
}

void McpPersisterTask::complete(McpToolResult const& result)
{
    if (!_completion) {
        return;
    }
    auto completion = std::move(*_completion);
    _completion.reset();
    completion(result);
}
