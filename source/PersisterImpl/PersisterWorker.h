#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "PersisterInterface/PersisterRequestState.h"

#include "Definitions.h"
#include "PersisterRequest.h"
#include "PersisterRequestError.h"
#include "PersisterRequestResult.h"

class _PersisterWorker
{
public:
    _PersisterWorker(SimulationFacade const& simulationFacade);

    void runThreadLoop();
    void shutdown();
    void restart();

    bool isBusy() const;
    std::optional<PersisterRequestState> getRequestState(PersisterRequestId const& id) const;

    void addRequest(PersisterRequest const& job);
    PersisterRequestResult fetchRequestResult(PersisterRequestId const& id);   
    PersisterRequestError fetchJobError(PersisterRequestId const& id);   

    std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId);

private:
    void processRequests(std::unique_lock<std::mutex>& lock);

    using PersisterRequestResultOrError = std::variant<PersisterRequestResult, PersisterRequestError>;
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, SaveSimulationRequest const& job);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, ReadSimulationRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, LoginRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, GetNetworkResourcesRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, DownloadNetworkResourceRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, UploadNetworkResourceRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, ReplaceNetworkResourceRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, GetUserNamesForEmojiRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, DeleteNetworkResourceRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, EditNetworkResourceRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, MoveNetworkResourceRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, ToggleReactionNetworkResourceRequest const& request);

    SimulationFacade _simulationFacade;

    std::atomic<bool> _isShutdown{false};

    mutable std::mutex _requestMutex;
    std::deque<PersisterRequest> _openRequests;
    std::deque<PersisterRequest> _inProgressRequests;
    std::deque<PersisterRequestResult> _finishedRequests;
    std::deque<PersisterRequestError> _requestErrors;

    std::condition_variable _conditionVariable;
};
