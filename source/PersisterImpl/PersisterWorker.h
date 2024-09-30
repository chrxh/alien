#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "PersisterInterface/PersisterJobState.h"

#include "Definitions.h"
#include "PersisterJob.h"
#include "PersisterJobError.h"
#include "PersisterJobResult.h"

class _PersisterWorker
{
public:
    _PersisterWorker(SimulationController const& simController);

    void runThreadLoop();
    void shutdown();

    bool isBusy() const;
    PersisterJobState getJobState(PersisterJobId const& id) const;
    std::variant<PersisterJobResult, PersisterJobError> fetchJobResult(PersisterJobId const& id);

    void addJob(PersisterJob const& job);

private:
    void processJobs(std::unique_lock<std::mutex>& lock);

    std::variant<PersisterJobResult, PersisterJobError> processSaveToDiscJob(std::unique_lock<std::mutex>& lock, SaveToDiscJob const& job);

    SimulationController _simController;

    std::atomic<bool> _isShutdown{false};

    mutable std::mutex _jobMutex;
    std::deque<PersisterJob> _openJobs;
    std::deque<PersisterJob> _inProgressJobs;
    std::deque<PersisterJobResult> _finishedJobs;
    std::deque<PersisterJobError> _errorJobs;

    std::condition_variable _conditionVariable;
};
