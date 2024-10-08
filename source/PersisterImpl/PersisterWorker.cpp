#include "PersisterWorker.h"

#include <algorithm>
#include <filesystem>

#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"

_PersisterWorker::_PersisterWorker(SimulationController const& simController)
    : _simController(simController)
{
}

void _PersisterWorker::runThreadLoop()
{
    std::unique_lock lock(_jobMutex);
    while (!_isShutdown.load()) {
        _conditionVariable.wait(lock);
        processJobs(lock);
    }
}

void _PersisterWorker::restart()
{
    _isShutdown = false;
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

PersisterRequestState _PersisterWorker::getJobState(PersisterRequestId const& id) const
{
    std::unique_lock uniqueLock(_jobMutex);

    if (std::ranges::find_if(_openJobs, [&](PersisterRequest const& request) { return request->getRequestId() == id; }) != _openJobs.end()) {
        return PersisterRequestState::InQueue;
    }
    if (std::ranges::find_if(_inProgressJobs, [&](PersisterRequest const& request) { return request->getRequestId() == id; }) != _inProgressJobs.end()) {
        return PersisterRequestState::InProgress;
    }
    if (std::ranges::find_if(_finishedJobs, [&](PersisterRequestResult const& request) { return request->getRequestId() == id; }) != _finishedJobs.end()) {
        return PersisterRequestState::Finished;
    }
    if (std::ranges::find_if(_jobErrors, [&](PersisterRequestError const& request) { return request->getRequestId() == id; }) != _jobErrors.end()) {
        return PersisterRequestState::Error;
    }
    THROW_NOT_IMPLEMENTED();
}

void _PersisterWorker::addRequest(PersisterRequest const& job)
{
    {
        std::unique_lock uniqueLock(_jobMutex);

        _openJobs.emplace_back(job);
    }
    _conditionVariable.notify_all();
}

PersisterRequestResult _PersisterWorker::fetchJobResult(PersisterRequestId const& id)
{
    std::unique_lock uniqueLock(_jobMutex);

    auto finishedJobsIter = std::ranges::find_if(_finishedJobs, [&](PersisterRequestResult const& job) { return job->getRequestId() == id; });
    if (finishedJobsIter != _finishedJobs.end()) {
        auto resultCopy = *finishedJobsIter;
        _finishedJobs.erase(finishedJobsIter);
        return resultCopy;
    }
    THROW_NOT_IMPLEMENTED();
}

PersisterRequestError _PersisterWorker::fetchJobError(PersisterRequestId const& id)
{
    std::unique_lock uniqueLock(_jobMutex);

    auto jobsErrorsIter = std::ranges::find_if(_jobErrors, [&](PersisterRequestError const& job) { return job->getRequestId() == id; });
    if (jobsErrorsIter != _jobErrors.end()) {
        auto resultCopy = *jobsErrorsIter;
        _jobErrors.erase(jobsErrorsIter);
        return resultCopy;
    }
    THROW_NOT_IMPLEMENTED();
}

std::vector<PersisterErrorInfo> _PersisterWorker::fetchAllErrorInfos(SenderId const& senderId)
{
    std::unique_lock lock(_jobMutex);

    std::vector<PersisterErrorInfo> result;
    std::deque<PersisterRequestError> filteredErrorJobs;
    for (auto const& errorJob : _jobErrors) {
        if (errorJob->getSenderId() == senderId) {
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

        auto request = _openJobs.front();
        _openJobs.pop_front();

        _inProgressJobs.push_back(request);

        std::variant<PersisterRequestResult, PersisterRequestError> processingResult;
        if (auto const& saveToFileJob = std::dynamic_pointer_cast<_SaveToFileRequest>(request)) {
            processingResult = processRequest(lock, saveToFileJob);
        }
        if (auto const& loadFromFileJob = std::dynamic_pointer_cast<_ReadFromFileRequest>(request)) {
            processingResult = processRequest(lock, loadFromFileJob);
        }
        auto inProgressJobsIter = std::ranges::find_if(
            _inProgressJobs, [&](PersisterRequest const& otherRequest) { return otherRequest->getRequestId() == request->getRequestId(); });
        _inProgressJobs.erase(inProgressJobsIter);

        if (std::holds_alternative<PersisterRequestResult>(processingResult)) {
            if (request->getSenderInfo().wishResultData) {
                _finishedJobs.emplace_back(std::get<PersisterRequestResult>(processingResult));
            }
        }
        if (std::holds_alternative<PersisterRequestError>(processingResult)) {
            if (request->getSenderInfo().wishErrorInfo) {
                _jobErrors.emplace_back(std::get<PersisterRequestError>(processingResult));
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

auto _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, SaveToFileRequest const& request) -> PersisterRequestResultOrError
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    DeserializedSimulation deserializedData;
    std::string simulationName;
    std::chrono::system_clock::time_point timePoint;
    try {
        simulationName = _simController->getSimulationName();
        timePoint = std::chrono::system_clock::now();
        deserializedData.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        deserializedData.auxiliaryData.realTime = _simController->getRealTime();
        deserializedData.auxiliaryData.zoom = requestData.zoom;
        deserializedData.auxiliaryData.center = requestData.center;
        deserializedData.auxiliaryData.generalSettings = _simController->getGeneralSettings();
        deserializedData.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
        deserializedData.statistics = _simController->getStatisticsHistory().getCopiedData();
        deserializedData.mainData = _simController->getClusteredSimulationData();
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because no valid data could be obtained from the GPU."});
    }

    try {
        SerializerService::serializeSimulationToFiles(requestData.filename, deserializedData);

        return std::make_shared<_SaveToFileRequestResult>(
            request->getRequestId(), SavedSimulationResultData{simulationName, deserializedData.auxiliaryData.timestep, timePoint});
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because an error occurred when serializing the data to the file."});
    }
}

auto _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, ReadFromFileRequest const& request) -> PersisterRequestResultOrError
{
    UnlockGuard unlockGuard(lock);

    try {
        auto const& requestData = request->getData();

        DeserializedSimulation deserializedData;
        if (!SerializerService::deserializeSimulationFromFiles(deserializedData, requestData.filename)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The selected file could not be opened."});
        }
        auto simulationName = std::filesystem::path(requestData.filename).stem().string();
        return std::make_shared<_ReadFromFileRequestResult>(request->getRequestId(), ReadSimulationResultData{simulationName, deserializedData});
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be loaded because an error occurred when deserializing the data from the file."});
    }
}
