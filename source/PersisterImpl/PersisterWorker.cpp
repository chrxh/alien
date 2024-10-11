#include "PersisterWorker.h"

#include <algorithm>
#include <filesystem>

#include "Base/LoggingService.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"
#include "Network/NetworkService.h"

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

    return !_openRequests.empty() || !_inProgressRequests.empty();
}

PersisterRequestState _PersisterWorker::getJobState(PersisterRequestId const& id) const
{
    std::unique_lock uniqueLock(_jobMutex);

    if (std::ranges::find_if(_openRequests, [&](PersisterRequest const& request) { return request->getRequestId() == id; }) != _openRequests.end()) {
        return PersisterRequestState::InQueue;
    }
    if (std::ranges::find_if(_inProgressRequests, [&](PersisterRequest const& request) { return request->getRequestId() == id; }) != _inProgressRequests.end()) {
        return PersisterRequestState::InProgress;
    }
    if (std::ranges::find_if(_finishedRequests, [&](PersisterRequestResult const& request) { return request->getRequestId() == id; }) != _finishedRequests.end()) {
        return PersisterRequestState::Finished;
    }
    if (std::ranges::find_if(_requestErrors, [&](PersisterRequestError const& request) { return request->getRequestId() == id; }) != _requestErrors.end()) {
        return PersisterRequestState::Error;
    }
    THROW_NOT_IMPLEMENTED();
}

void _PersisterWorker::addRequest(PersisterRequest const& job)
{
    {
        std::unique_lock uniqueLock(_jobMutex);

        _openRequests.emplace_back(job);
    }
    _conditionVariable.notify_all();
}

PersisterRequestResult _PersisterWorker::fetchRequestResult(PersisterRequestId const& id)
{
    std::unique_lock uniqueLock(_jobMutex);

    auto finishedJobsIter = std::ranges::find_if(_finishedRequests, [&](PersisterRequestResult const& job) { return job->getRequestId() == id; });
    if (finishedJobsIter != _finishedRequests.end()) {
        auto resultCopy = *finishedJobsIter;
        _finishedRequests.erase(finishedJobsIter);
        return resultCopy;
    }
    THROW_NOT_IMPLEMENTED();
}

PersisterRequestError _PersisterWorker::fetchJobError(PersisterRequestId const& id)
{
    std::unique_lock uniqueLock(_jobMutex);

    auto jobsErrorsIter = std::ranges::find_if(_requestErrors, [&](PersisterRequestError const& job) { return job->getRequestId() == id; });
    if (jobsErrorsIter != _requestErrors.end()) {
        auto resultCopy = *jobsErrorsIter;
        _requestErrors.erase(jobsErrorsIter);
        return resultCopy;
    }
    THROW_NOT_IMPLEMENTED();
}

std::vector<PersisterErrorInfo> _PersisterWorker::fetchAllErrorInfos(SenderId const& senderId)
{
    std::unique_lock lock(_jobMutex);

    std::vector<PersisterErrorInfo> result;
    std::deque<PersisterRequestError> filteredErrorJobs;
    for (auto const& errorJob : _requestErrors) {
        if (errorJob->getSenderId() == senderId) {
            result.emplace_back(errorJob->getErrorInfo());
        } else {
            filteredErrorJobs.emplace_back(errorJob);
        }
    }
    _requestErrors = filteredErrorJobs;
    return result;
}

void _PersisterWorker::processJobs(std::unique_lock<std::mutex>& lock)
{
    if (_openRequests.empty()) {
        return;
    }

    while (!_openRequests.empty()) {

        auto request = _openRequests.front();
        _openRequests.pop_front();

        _inProgressRequests.push_back(request);

        std::variant<PersisterRequestResult, PersisterRequestError> processingResult;
        if (auto const& concreteRequest = std::dynamic_pointer_cast<_SaveToFileRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        }
        if (auto const& concreteRequest = std::dynamic_pointer_cast<_ReadFromFileRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        }
        if (auto const& concreteRequest = std::dynamic_pointer_cast<_LoginRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        }
        if (auto const& concreteRequest = std::dynamic_pointer_cast<_GetNetworkResourcesRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        }
        if (auto const& concreteRequest = std::dynamic_pointer_cast<_DownloadNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        }
        auto inProgressJobsIter = std::ranges::find_if(
            _inProgressRequests, [&](PersisterRequest const& otherRequest) { return otherRequest->getRequestId() == request->getRequestId(); });
        _inProgressRequests.erase(inProgressJobsIter);

        if (std::holds_alternative<PersisterRequestResult>(processingResult)) {
            if (request->getSenderInfo().wishResultData) {
                _finishedRequests.emplace_back(std::get<PersisterRequestResult>(processingResult));
            }
        }
        if (std::holds_alternative<PersisterRequestError>(processingResult)) {
            if (request->getSenderInfo().wishErrorInfo) {
                _requestErrors.emplace_back(std::get<PersisterRequestError>(processingResult));
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

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, LoginRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    LoginErrorCode errorCode;
    if (!NetworkService::login(errorCode, requestData.userName, requestData.password, requestData.userInfo)) {
        if (errorCode != LoginErrorCode_UnknownUser) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(),
                request->getSenderInfo().senderId,
                PersisterErrorInfo{"Login failed."});
        }
        if (errorCode == LoginErrorCode_UnknownUser) {
            return std::make_shared<_LoginRequestResult>(request->getRequestId(), LoginResultData{.unknownUser = true});
        }
    }
    return std::make_shared<_LoginRequestResult>(request->getRequestId(), LoginResultData{.unknownUser = false});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, GetNetworkResourcesRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    NetworkService::refreshLogin();

    GetNetworkResourcesResultData data;

    auto withRetry = true;
    bool success = NetworkService::getNetworkResources(data.resourceTOs, withRetry);
    if (success) {
        success &= NetworkService::getUserList(data.userTOs, withRetry);
    }
    if (success && NetworkService::getLoggedInUserName()) {
        success &= NetworkService::getEmojiTypeByResourceId(data.emojiTypeByResourceId);
    }

    if (!success) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"Failed to retrieve browser data. Please try again."});
    }

    return std::make_shared<_GetNetworkResourcesRequestResult>(request->getRequestId(), data);
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(
    std::unique_lock<std::mutex>& lock,
    DownloadNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();
    DownloadNetworkResourceResultData resultData;
    resultData.resourceName = requestData.resourceName;
    resultData.resourceVersion = requestData.resourceVersion;
    resultData.resourceType = requestData.resourceType;

    std::string dataTypeString = requestData.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
    std::optional<DeserializedSimulation> cachedSimulation;
    if (requestData.resourceType == NetworkResourceType_Simulation) {
        cachedSimulation = requestData.downloadCache->find(requestData.resourceId);
    }
    SerializedSimulation serializedSim;
    if (!cachedSimulation.has_value()) {
        if (!NetworkService::downloadResource(serializedSim.mainData, serializedSim.auxiliaryData, serializedSim.statistics, requestData.resourceId)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Failed to download " + dataTypeString + "."});
        }
    }

    if (requestData.resourceType == NetworkResourceType_Simulation) {
        DeserializedSimulation deserializedSimulation;
        if (!cachedSimulation.has_value()) {
            if (!SerializerService::deserializeSimulationFromStrings(deserializedSimulation, serializedSim)) {
                return std::make_shared<_PersisterRequestError>(
                    request->getRequestId(),
                    request->getSenderInfo().senderId,
                    PersisterErrorInfo{"Failed to load simulation. Your program version may not match."});
            }
            requestData.downloadCache->insertOrAssign(requestData.resourceId, deserializedSimulation);
        } else {
            log(Priority::Important, "browser: get resource with id=" + requestData.resourceId + " from simulation cache");
            std::swap(deserializedSimulation, *cachedSimulation);
            NetworkService::incDownloadCounter(requestData.resourceId);
        }
        resultData.resourceData.emplace<DeserializedSimulation>(std::move(deserializedSimulation));
    } else {
        std::vector<uint8_t> genome;
        if (!SerializerService::deserializeGenomeFromString(genome, serializedSim.mainData)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(),
                request->getSenderInfo().senderId,
                PersisterErrorInfo{"Failed to load genome. Your program version may not match."});
        }
        resultData.resourceData = GenomeDescriptionService::convertBytesToDescription(genome);
    }

    return std::make_shared<_DownloadNetworkResourceRequestResult>(request->getRequestId(), resultData);
}
