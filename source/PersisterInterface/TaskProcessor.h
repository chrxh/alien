#pragma once

#include <functional>
#include <vector>

#include "PersisterFacade.h"
#include "PersisterRequestId.h"
#include "Definitions.h"

class _TaskProcessor
{
public:
    static TaskProcessor createTaskProcessor(PersisterFacade const& persisterFacade);

    void executeTask(
        std::function<PersisterRequestId(SenderId const&)> const& requestFunc,
        std::function<void(PersisterRequestId const&)> const& finishFunc,
        std::function<void(std::vector<PersisterErrorInfo> const&)> const& errorFunc);

    bool pendingTasks() const;

    void process();

private:
    _TaskProcessor(PersisterFacade const& persisterFacade, std::string const& senderId);

    std::function<void(PersisterRequestId const&)> _finishFunc;
    std::function<void(std::vector<PersisterErrorInfo> const&)> _errorFunc;
    std::string _senderId;

    PersisterFacade _persisterFacade;
    std::vector<PersisterRequestId> _pendingRequestIds;
};
