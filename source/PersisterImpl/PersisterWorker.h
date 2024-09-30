#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "PersisterInterface/PersisterJobState.h"

#include "Definitions.h"
#include "PersisterJob.h"
#include "PersisterJobResult.h"

    class _PersisterWorker
{
public:
    _PersisterWorker(SimulationController const& simController);

    void runThreadLoop();
    void shutdown();

    PersisterJobState getJobState(PersisterJobId const& id) const;

    void addJob(PersisterJob const& job);

private:
    void processJobs(std::unique_lock<std::mutex>& lock);

    PersisterJobResult processSaveToDiscJob(std::unique_lock<std::mutex>& lock, SaveToDiscJob const& job);

    SimulationController _simController;

    std::atomic<bool> _isShutdown{false};

    mutable std::mutex _jobMutex;
    std::deque<PersisterJob> _openJobs;
    std::vector<PersisterJob> _inProgressJobs;
    std::vector<PersisterJobResult> _finishedJobs;

    int _idCount = 0;
    std::condition_variable _conditionVariable;
};
