#include "TaskProcessor.h"

TaskProcessor _TaskProcessor::createTaskProcessor(PersisterController const& persisterController)
{
    static auto counter = 0;
    ++counter;
    return std::shared_ptr<_TaskProcessor>(new _TaskProcessor(persisterController, "Processor" + std::to_string(counter)));
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
        auto state = _persisterController->getRequestState(requestId);
        if (state == PersisterRequestState::Finished) {
            _finishFunc(requestId);
        }
        if (state == PersisterRequestState::InQueue || state == PersisterRequestState::InProgress) {
            newRequestIds.emplace_back(requestId);
        }
    }
    _pendingRequestIds = newRequestIds;

    auto criticalErrors = _persisterController->fetchAllErrorInfos(SenderId{_senderId});
    if (!criticalErrors.empty()) {
        _errorFunc(criticalErrors);
    }
}

_TaskProcessor::_TaskProcessor(PersisterController const& persisterController, std::string const& senderId)
    : _persisterController(persisterController)
    , _senderId(senderId)
{}
