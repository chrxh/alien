#include "PersisterWorker.h"

#include <algorithm>

#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"

_PersisterWorker::_PersisterWorker(SimulationController const& simController)
    : _simController(simController)
{
}

void _PersisterWorker::runThreadLoop()
{
    try {
        std::unique_lock lock(_jobMutex);
        while (!_isShutdown.load()) {
            _conditionVariable.wait(lock);
            processJobs(lock);
        }
    } catch (std::exception const&) {
        //#TODO
    }
}

void _PersisterWorker::shutdown()
{
    _isShutdown = true;
    _conditionVariable.notify_all();
}

bool _PersisterWorker::isBusy() const
{
    std::unique_lock uniqueLock(_jobMutex);

    return !_openJobs.empty() || !_inProgressJobs.empty();
}

PersisterJobState _PersisterWorker::getJobState(PersisterJobId const& id) const
{
    std::unique_lock uniqueLock(_jobMutex);

    if (std::ranges::find_if(_openJobs, [&](PersisterJob const& job) { return job->getId() == id; }) != _openJobs.end()) {
        return PersisterJobState::InQueue;
    }
    if (std::ranges::find_if(_inProgressJobs, [&](PersisterJob const& job) { return job->getId() == id; }) != _inProgressJobs.end()) {
        return PersisterJobState::InProgress;
    }
    if (std::ranges::find_if(_finishedJobs, [&](PersisterJobResult const& job) { return job->getId() == id; }) != _finishedJobs.end()) {
        return PersisterJobState::Finished;
    }
    if (std::ranges::find_if(_jobErrors, [&](PersisterJobError const& job) { return job->getId() == id; }) != _jobErrors.end()) {
        return PersisterJobState::Error;
    }
    THROW_NOT_IMPLEMENTED();
}

void _PersisterWorker::addJob(PersisterJob const& job)
{
    {
        std::unique_lock uniqueLock(_jobMutex);

        _openJobs.emplace_back(job);
    }
    _conditionVariable.notify_all();
}

std::variant<PersisterJobResult, PersisterJobError> _PersisterWorker::fetchJobResult(PersisterJobId const& id)
{
    std::unique_lock uniqueLock(_jobMutex);

    auto finishedJobsIter = std::ranges::find_if(_finishedJobs, [&](PersisterJobResult const& job) { return job->getId() == id; });
    if (finishedJobsIter != _finishedJobs.end()) {
        auto resultCopy = *finishedJobsIter;
        _finishedJobs.erase(finishedJobsIter);
        return resultCopy;
    }

    auto jobsErrorsIter = std::ranges::find_if(_jobErrors, [&](PersisterJobError const& job) { return job->getId() == id; });
    if (jobsErrorsIter != _jobErrors.end()) {
        auto resultCopy = *jobsErrorsIter;
        _jobErrors.erase(jobsErrorsIter);
        return resultCopy;
    }
    THROW_NOT_IMPLEMENTED();
}

std::vector<PersisterErrorInfo> _PersisterWorker::fetchCriticalErrorInfos()
{
    std::unique_lock lock(_jobMutex);

    std::vector<PersisterErrorInfo> result;
    std::deque<PersisterJobError> filteredErrorJobs;
    for (auto const& errorJob : _jobErrors) {
        if (errorJob->isCritical()) {
            result.emplace_back(errorJob->getErrorInfo());
        } else {
            filteredErrorJobs.emplace_back(errorJob);
        }
    }
    _jobErrors = filteredErrorJobs;
    return result;
}

void _PersisterWorker::processJobs(std::unique_lock<std::mutex>& lock)
{
    if (_openJobs.empty()) {
        return;
    }

    while (!_openJobs.empty()) {

        auto job = _openJobs.front();
        _openJobs.pop_front();

        if (auto const& saveToDiscJob = std::dynamic_pointer_cast<_SaveToDiscJob>(job)) {
            _inProgressJobs.push_back(job);
            auto processingResult = processSaveToDiscJob(lock, saveToDiscJob);
            auto inProgressJobsIter = std::ranges::find_if(_inProgressJobs, [&](PersisterJob const& otherJob) { return otherJob->getId() == job->getId(); });
            _inProgressJobs.erase(inProgressJobsIter);

            if (std::holds_alternative<PersisterJobResult>(processingResult)) {
                _finishedJobs.emplace_back(std::get<PersisterJobResult>(processingResult));
            }
            if (std::holds_alternative<PersisterJobError>(processingResult)) {
                _jobErrors.emplace_back(std::get<PersisterJobError>(processingResult));
            }
        }

    }
}

namespace
{
    class UnlockGuard
    {
    public:
        UnlockGuard(std::unique_lock<std::mutex>& lock) : _lock(lock) { _lock.unlock(); }
        ~UnlockGuard() { _lock.lock(); }

    private:
        std::unique_lock<std::mutex>& _lock;
    };
}

std::variant<PersisterJobResult, PersisterJobError> _PersisterWorker::processSaveToDiscJob(std::unique_lock<std::mutex>& lock, SaveToDiscJob const& job)
{
    UnlockGuard unlockGuard(lock);

    DeserializedSimulation deserializedData;
    try {
        deserializedData.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        deserializedData.auxiliaryData.realTime = _simController->getRealTime();
        deserializedData.auxiliaryData.zoom = job->getZoom();
        deserializedData.auxiliaryData.center = job->getCenter();
        deserializedData.auxiliaryData.generalSettings = _simController->getGeneralSettings();
        deserializedData.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
        deserializedData.statistics = _simController->getStatisticsHistory().getCopiedData();
        deserializedData.mainData = _simController->getClusteredSimulationData();
    } catch (std::runtime_error const&) {
        return std::make_shared<_PersisterJobError>(
            job->getId(), job->isCritical(), PersisterErrorInfo{"The simulation could not be saved because no valid data could be obtained from the GPU."});
    }

    try {
        SerializerService::serializeSimulationToFiles(job->getFilename(), deserializedData);
    } catch (std::runtime_error const&) {
        return std::make_shared<_PersisterJobError>(
            job->getId(),
            job->isCritical(),
            PersisterErrorInfo{"The simulation could not be saved because an error occurred when serializing the data to the file."});
    }

    return std::make_shared<_SaveToDiscJobResult>(job->getId(), deserializedData.auxiliaryData.timestep, deserializedData.auxiliaryData.realTime);
}
