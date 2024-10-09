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
    _PersisterWorker(SimulationController const& simController);

    void runThreadLoop();
    void shutdown();
    void restart();

    bool isBusy() const;
    PersisterRequestState getJobState(PersisterRequestId const& id) const;

    void addRequest(PersisterRequest const& job);
    PersisterRequestResult fetchJobResult(PersisterRequestId const& id);   
    PersisterRequestError fetchJobError(PersisterRequestId const& id);   

    std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId);

private:
    void processJobs(std::unique_lock<std::mutex>& lock);

    using PersisterRequestResultOrError = std::variant<PersisterRequestResult, PersisterRequestError>;
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, SaveToFileRequest const& job);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, ReadFromFileRequest const& request);
    PersisterRequestResultOrError processRequest(std::unique_lock<std::mutex>& lock, GetNetworkResourcesRequest const& request);

    SimulationController _simController;

    std::atomic<bool> _isShutdown{false};

    mutable std::mutex _jobMutex;
    std::deque<PersisterRequest> _openJobs;
    std::deque<PersisterRequest> _inProgressJobs;
    std::deque<PersisterRequestResult> _finishedJobs;
    std::deque<PersisterRequestError> _jobErrors;

    std::condition_variable _conditionVariable;
};
