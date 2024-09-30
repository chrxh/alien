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

PersisterRequestState _PersisterWorker::getJobState(PersisterRequestId const& id) const
{
    std::unique_lock uniqueLock(_jobMutex);

    if (std::ranges::find_if(_openJobs, [&](PersisterRequest const& job) { return job->getId() == id; }) != _openJobs.end()) {
        return PersisterRequestState::InQueue;
    }
    if (std::ranges::find_if(_inProgressJobs, [&](PersisterRequest const& job) { return job->getId() == id; }) != _inProgressJobs.end()) {
        return PersisterRequestState::InProgress;
    }
    if (std::ranges::find_if(_finishedJobs, [&](PersisterRequestResult const& job) { return job->getRequestId() == id; }) != _finishedJobs.end()) {
        return PersisterRequestState::Finished;
    }
    if (std::ranges::find_if(_jobErrors, [&](PersisterRequestError const& job) { return job->getId() == id; }) != _jobErrors.end()) {
        return PersisterRequestState::Error;
    }
    THROW_NOT_IMPLEMENTED();
}

void _PersisterWorker::addJob(PersisterRequest const& job)
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

    auto jobsErrorsIter = std::ranges::find_if(_jobErrors, [&](PersisterRequestError const& job) { return job->getId() == id; });
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

        auto job = _openJobs.front();
        _openJobs.pop_front();

        _inProgressJobs.push_back(job);

        std::variant<PersisterRequestResult, PersisterRequestError> processingResult;
        if (auto const& saveToFileJob = std::dynamic_pointer_cast<_SaveToFileJob>(job)) {
            processingResult = processJob(lock, saveToFileJob);
        }
        if (auto const& loadFromFileJob = std::dynamic_pointer_cast<_LoadFromFileJob>(job)) {
            processingResult = processJob(lock, loadFromFileJob);
        }
        auto inProgressJobsIter = std::ranges::find_if(_inProgressJobs, [&](PersisterRequest const& otherJob) { return otherJob->getId() == job->getId(); });
        _inProgressJobs.erase(inProgressJobsIter);

        if (std::holds_alternative<PersisterRequestResult>(processingResult)) {
            if (job->getSenderInfo().wishResultData) {
                _finishedJobs.emplace_back(std::get<PersisterRequestResult>(processingResult));
            }
        }
        if (std::holds_alternative<PersisterRequestError>(processingResult)) {
            if (job->getSenderInfo().wishErrorInfo) {
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
auto _PersisterWorker::processJob(std::unique_lock<std::mutex>& lock, SaveToFileJob const& job) -> PersisterRequestResultOrError
{
    UnlockGuard unlockGuard(lock);

    DeserializedSimulation deserializedData;
    std::string simulationName;
    std::chrono::system_clock::time_point timePoint;
    try {
        simulationName = _simController->getSimulationName();
        timePoint = std::chrono::system_clock::now();
        deserializedData.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        deserializedData.auxiliaryData.realTime = _simController->getRealTime();
        deserializedData.auxiliaryData.zoom = job->getZoom();
        deserializedData.auxiliaryData.center = job->getCenter();
        deserializedData.auxiliaryData.generalSettings = _simController->getGeneralSettings();
        deserializedData.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
        deserializedData.statistics = _simController->getStatisticsHistory().getCopiedData();
        deserializedData.mainData = _simController->getClusteredSimulationData();
    } catch (std::runtime_error const&) {
        return std::make_shared<_PersisterRequestError>(
            job->getId(),
            job->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because no valid data could be obtained from the GPU."});
    }

    try {
        SerializerService::serializeSimulationToFiles(job->getFilename(), deserializedData);
    } catch (std::runtime_error const&) {
        return std::make_shared<_PersisterRequestError>(
            job->getId(),
            job->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because an error occurred when serializing the data to the file."});
    }

    return std::make_shared<_SaveToFileJobResult>(job->getId(), simulationName, deserializedData.auxiliaryData.timestep, timePoint);
}

auto _PersisterWorker::processJob(std::unique_lock<std::mutex>& lock, LoadFromFileJob const& job) -> PersisterRequestResultOrError
{
    return PersisterRequestError{};
    //DeserializedSimulation deserializedData;
    //if (SerializerService::deserializeSimulationFromFiles(deserializedData, firstFilename.string())) {
    //    _simController->closeSimulation();

    //    std::optional<std::string> errorMessage;
    //    try {
    //        _simController->newSimulation(
    //            firstFilename.stem().string(),
    //            deserializedData.auxiliaryData.timestep,
    //            deserializedData.auxiliaryData.generalSettings,
    //            deserializedData.auxiliaryData.simulationParameters);
    //        _simController->setClusteredSimulationData(deserializedData.mainData);
    //        _simController->setStatisticsHistory(deserializedData.statistics);
    //        _simController->setRealTime(deserializedData.auxiliaryData.realTime);
    //    } catch (CudaMemoryAllocationException const& exception) {
    //        errorMessage = exception.what();
    //    } catch (...) {
    //        errorMessage = "Failed to load simulation.";
    //    }

    //    if (errorMessage) {
    //        showMessage("Error", *errorMessage);
    //        _simController->closeSimulation();
    //        _simController->newSimulation(
    //            std::nullopt,
    //            deserializedData.auxiliaryData.timestep,
    //            deserializedData.auxiliaryData.generalSettings,
    //            deserializedData.auxiliaryData.simulationParameters);
    //    }

    //    Viewport::setCenterInWorldPos(deserializedData.auxiliaryData.center);
    //    Viewport::setZoomFactor(deserializedData.auxiliaryData.zoom);
    //    _temporalControlWindow->onSnapshot();
    //    printOverlayMessage(firstFilename.filename().string());
    //} else {
    //    showMessage("Open simulation", "The selected file could not be opened.");
    //}
}
