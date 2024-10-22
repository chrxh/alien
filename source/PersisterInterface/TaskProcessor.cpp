#include "TaskProcessor.h"

TaskProcessor _TaskProcessor::createTaskProcessor(PersisterFacade const& persisterFacade)
{
    static auto counter = 0;
    ++counter;
    return std::shared_ptr<_TaskProcessor>(new _TaskProcessor(persisterFacade, "Processor" + std::to_string(counter)));
}

void _TaskProcessor::executeTask(
    std::function<PersisterRequestId(SenderId const&)> const& requestFunc,
    std::function<void(PersisterRequestId const&)> const& finishFunc,
    std::function<void(std::vector<PersisterErrorInfo> const&)> const& errorFunc)
{
    _pendingRequestIds.emplace_back(requestFunc(SenderId{_senderId}));
    _finishFunc = finishFunc;
    _errorFunc = errorFunc;
}

bool _TaskProcessor::pendingTasks() const
{
    return !_pendingRequestIds.empty();
}

void _TaskProcessor::process()
{
    if (_pendingRequestIds.empty()) {
        return;
    }
    std::vector<PersisterRequestId> newRequestIds;
    for (auto const& requestId : _pendingRequestIds) {
        if (auto state = _persisterFacade->getRequestState(requestId)) {
            if (state.value() == PersisterRequestState::Finished) {
                _finishFunc(requestId);
            }
            if (state.value() == PersisterRequestState::InQueue || state.value() == PersisterRequestState::InProgress) {
                newRequestIds.emplace_back(requestId);
            }
        }
    }
    _pendingRequestIds = newRequestIds;

    auto criticalErrors = _persisterFacade->fetchAllErrorInfos(SenderId{_senderId});
    if (!criticalErrors.empty()) {
        _errorFunc(criticalErrors);
    }
}

_TaskProcessor::_TaskProcessor(PersisterFacade const& persisterFacade, std::string const& senderId)
    : _persisterFacade(persisterFacade)
    , _senderId(senderId)
{}
