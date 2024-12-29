#include "PersisterWorker.h"

#include <algorithm>
#include <filesystem>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/LoggingService.h"
#include "Base/StringHelper.h"
#include "Base/UnlockGuard.h"
#include "PersisterInterface/SerializerService.h"
#include "PersisterInterface/PersisterRequestResult.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "Network/NetworkService.h"

_PersisterWorker::_PersisterWorker(SimulationFacade const& simulationFacade)
    : _simulationFacade(simulationFacade)
{
}

void _PersisterWorker::runThreadLoop()
{
    std::unique_lock lock(_requestMutex);
    while (!_isShutdown.load()) {
        _conditionVariable.wait(lock);
        processRequests(lock);
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
    std::unique_lock uniqueLock(_requestMutex);

    return !_openRequests.empty() || !_inProgressRequests.empty();
}

std::optional<PersisterRequestState> _PersisterWorker::getRequestState(PersisterRequestId const& id) const
{
    std::unique_lock uniqueLock(_requestMutex);

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
    return std::nullopt;
}

void _PersisterWorker::addRequest(PersisterRequest const& job)
{
    {
        std::unique_lock uniqueLock(_requestMutex);

        _openRequests.emplace_back(job);
    }
    _conditionVariable.notify_all();
}

PersisterRequestResult _PersisterWorker::fetchRequestResult(PersisterRequestId const& id)
{
    std::unique_lock uniqueLock(_requestMutex);

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
    std::unique_lock uniqueLock(_requestMutex);

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
    std::unique_lock lock(_requestMutex);

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

void _PersisterWorker::processRequests(std::unique_lock<std::mutex>& lock)
{
    if (_openRequests.empty()) {
        return;
    }

    while (!_openRequests.empty()) {

        auto request = _openRequests.front();
        _openRequests.pop_front();

        _inProgressRequests.push_back(request);

        std::variant<PersisterRequestResult, PersisterRequestError> processingResult;
        if (auto const& concreteRequest = std::dynamic_pointer_cast<_SaveSimulationRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_ReadSimulationRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_LoginRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_GetNetworkResourcesRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_DownloadNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_UploadNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_ReplaceNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_GetUserNamesForEmojiRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_DeleteNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_EditNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_MoveNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_ToggleReactionNetworkResourceRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_GetPeakSimulationRequest>(request)) {
            processingResult = processRequest(lock, concreteRequest);
        } else if (auto const& concreteRequest = std::dynamic_pointer_cast<_SaveDeserializedSimulationRequest>(request)) {
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
    std::filesystem::path generateFilename(std::filesystem::path const& directory, uint64_t timestep)
    {
        std::filesystem::path result;
        int i = 0;
        do {
            auto postfix = i == 0 ? std::string() : "-" + std::to_string(i);
            result = directory / ("save_" + StringHelper::format(timestep, '_') + postfix + ".sim");
            ++i;
        } while (std::filesystem::exists(result) && i < 100);
        return result;
    }
}

auto _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, SaveSimulationRequest const& request) -> PersisterRequestResultOrError
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    DeserializedSimulation deserializedData;
    std::chrono::system_clock::time_point timestamp;

    try {
        timestamp = std::chrono::system_clock::now();
        deserializedData.statistics = _simulationFacade->getStatisticsHistory().getCopiedData();
        deserializedData.auxiliaryData.realTime = _simulationFacade->getRealTime();
        deserializedData.auxiliaryData.zoom = requestData.zoom;
        deserializedData.auxiliaryData.center = requestData.center;
        deserializedData.auxiliaryData.generalSettings = _simulationFacade->getGeneralSettings();
        deserializedData.auxiliaryData.simulationParameters = _simulationFacade->getSimulationParameters();
        deserializedData.auxiliaryData.timestep = static_cast<uint32_t>(_simulationFacade->getCurrentTimestep());
        deserializedData.mainData = _simulationFacade->getClusteredSimulationData();
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because no valid data could be obtained from the GPU."});
    }

    try {
        auto filename = requestData.filename;
        if (requestData.generateNameFromTimestep) {
            filename = generateFilename(filename, deserializedData.auxiliaryData.timestep);
        }
        if (!SerializerService::get().serializeSimulationToFiles(filename, deserializedData)) {
            throw std::runtime_error("Error");
        }

        return std::make_shared<_SaveSimulationRequestResult>(
            request->getRequestId(),
            SaveSimulationResultData{
                .filename = filename,
                .projectName = deserializedData.auxiliaryData.simulationParameters.projectName,
                .timestep = deserializedData.auxiliaryData.timestep,
                .timestamp = timestamp});
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because an error occurred when writing the data to the specified file."});
    }
}

auto _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, ReadSimulationRequest const& request) -> PersisterRequestResultOrError
{
    UnlockGuard unlockGuard(lock);

    try {
        auto const& requestData = request->getData();

        DeserializedSimulation deserializedData;
        if (!SerializerService::get().deserializeSimulationFromFiles(deserializedData, requestData.filename)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The selected file could not be opened."});
        }
        return std::make_shared<_ReadSimulationRequestResult>(
            request->getRequestId(), ReadSimulationResultData{std::filesystem::path(requestData.filename).filename(), deserializedData});
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
    if (!NetworkService::get().login(errorCode, requestData.userName, requestData.password, requestData.userInfo)) {
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

    NetworkService::get().refreshLogin();

    GetNetworkResourcesResultData data;

    auto withRetry = true;
    bool success = NetworkService::get().getNetworkResources(data.resourceTOs, withRetry);
    if (success) {
        success &= NetworkService::get().getUserList(data.userTOs, withRetry);
    }
    if (success && NetworkService::get().getLoggedInUserName()) {
        success &= NetworkService::get().getEmojiTypeByResourceId(data.emojiTypeByResourceId);
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
        if (!NetworkService::get().downloadResource(serializedSim.mainData, serializedSim.auxiliaryData, serializedSim.statistics, requestData.resourceId)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Failed to download " + dataTypeString + "."});
        }
    }

    if (requestData.resourceType == NetworkResourceType_Simulation) {
        DeserializedSimulation deserializedSimulation;
        if (!cachedSimulation.has_value()) {
            if (!SerializerService::get().deserializeSimulationFromStrings(deserializedSimulation, serializedSim)) {
                return std::make_shared<_PersisterRequestError>(
                    request->getRequestId(),
                    request->getSenderInfo().senderId,
                    PersisterErrorInfo{"Failed to load simulation. Your program version may not match."});
            }
            requestData.downloadCache->insertOrAssign(requestData.resourceId, deserializedSimulation);
        } else {
            log(Priority::Important, "browser: get resource with id=" + requestData.resourceId + " from simulation cache");
            std::swap(deserializedSimulation, *cachedSimulation);
            NetworkService::get().incDownloadCounter(requestData.resourceId);
        }
        resultData.resourceData.emplace<DeserializedSimulation>(std::move(deserializedSimulation));
    } else {
        std::vector<uint8_t> genome;
        if (!SerializerService::get().deserializeGenomeFromString(genome, serializedSim.mainData)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(),
                request->getSenderInfo().senderId,
                PersisterErrorInfo{"Failed to load genome. Your program version may not match."});
        }
        resultData.resourceData = GenomeDescriptionService::get().convertBytesToDescription(genome);
    }

    return std::make_shared<_DownloadNetworkResourceRequestResult>(request->getRequestId(), resultData);
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(
    std::unique_lock<std::mutex>& lock,
    UploadNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();
    DownloadNetworkResourceResultData resultData;

    std::string mainData;
    std::string settings;
    std::string statistics;
    IntVector2D size;
    int numObjects = 0;

    auto resourceType = std::holds_alternative<UploadNetworkResourceRequestData::SimulationData>(requestData.data) ? NetworkResourceType_Simulation
                                                                                                                   : NetworkResourceType_Genome;
    DeserializedSimulation deserializedSim;
    if (resourceType == NetworkResourceType_Simulation) {
        try {
            auto simulationData = std::get<UploadNetworkResourceRequestData::SimulationData>(requestData.data);
            deserializedSim.auxiliaryData.timestep = static_cast<uint32_t>(_simulationFacade->getCurrentTimestep());
            deserializedSim.auxiliaryData.realTime = _simulationFacade->getRealTime();
            deserializedSim.auxiliaryData.zoom = simulationData.zoom;
            deserializedSim.auxiliaryData.center = simulationData.center;
            deserializedSim.auxiliaryData.generalSettings = _simulationFacade->getGeneralSettings();
            deserializedSim.auxiliaryData.simulationParameters = _simulationFacade->getSimulationParameters();
            deserializedSim.statistics = _simulationFacade->getStatisticsHistory().getCopiedData();
            deserializedSim.mainData = _simulationFacade->getClusteredSimulationData();
        } catch (...) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(),
                request->getSenderInfo().senderId,
                PersisterErrorInfo{"The simulation could not be uploaded because no valid data could be obtained from the GPU."});
        }

        SerializedSimulation serializedSim;
        if (!SerializerService::get().serializeSimulationToStrings(serializedSim, deserializedSim)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(),
                request->getSenderInfo().senderId,
                PersisterErrorInfo{"The simulation could not be serialized for uploading."});
        }
        mainData = serializedSim.mainData;
        settings = serializedSim.auxiliaryData;
        statistics = serializedSim.statistics;
        size = {deserializedSim.auxiliaryData.generalSettings.worldSizeX, deserializedSim.auxiliaryData.generalSettings.worldSizeY};
        numObjects = deserializedSim.mainData.getNumberOfCellAndParticles();
    } else {
        auto genome = std::get<UploadNetworkResourceRequestData::GenomeData>(requestData.data).description;
        if (genome.cells.empty()) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The is no valid genome for uploading selected."});
        }
        auto genomeData = GenomeDescriptionService::get().convertDescriptionToBytes(genome);
        numObjects = GenomeDescriptionService::get().getNumNodesRecursively(genomeData, true);

        if (!SerializerService::get().serializeGenomeToString(mainData, genomeData)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The genome could not be serialized for uploading."});
        }
    }

    std::string resourceId;
    if (!NetworkService::get().uploadResource(
            resourceId,
            requestData.folderName + requestData.resourceWithoutFolderName,
            requestData.resourceDescription,
            size,
            numObjects,
            mainData,
            settings,
            statistics,
            resourceType,
            requestData.workspaceType)) {
        std::string dataTypeString = resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{
                "Failed to upload " + dataTypeString
                + ".\n\nPossible reasons:\n\n" ICON_FA_CHEVRON_RIGHT " The server is not reachable.\n\n" ICON_FA_CHEVRON_RIGHT
                  " The total size of your uploads exceeds the allowed storage limit."});
    }
    if (resourceType == NetworkResourceType_Simulation) {
        requestData.downloadCache->insertOrAssign(resourceId, deserializedSim);
    }

    return std::make_shared<_UploadNetworkResourceRequestResult>(request->getRequestId(), UploadNetworkResourceResultData{});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(
    std::unique_lock<std::mutex>& lock,
    ReplaceNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    auto resourceType = std::holds_alternative<ReplaceNetworkResourceRequestData::SimulationData>(requestData.data) ? NetworkResourceType_Simulation
                                                                                                                   : NetworkResourceType_Genome;
    std::string mainData;
    std::string settings;
    std::string statistics;
    IntVector2D worldSize;
    int numObjects = 0;

    DeserializedSimulation deserializedSim;
    if (resourceType == NetworkResourceType_Simulation) {
        try {
            auto simulationData = std::get<ReplaceNetworkResourceRequestData::SimulationData>(requestData.data);
            deserializedSim.auxiliaryData.timestep = static_cast<uint32_t>(_simulationFacade->getCurrentTimestep());
            deserializedSim.auxiliaryData.realTime = _simulationFacade->getRealTime();
            deserializedSim.auxiliaryData.zoom = simulationData.zoom;
            deserializedSim.auxiliaryData.center = simulationData.center;
            deserializedSim.auxiliaryData.generalSettings = _simulationFacade->getGeneralSettings();
            deserializedSim.auxiliaryData.simulationParameters = _simulationFacade->getSimulationParameters();
            deserializedSim.statistics = _simulationFacade->getStatisticsHistory().getCopiedData();
            deserializedSim.mainData = _simulationFacade->getClusteredSimulationData();
        } catch (...) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(),
                request->getSenderInfo().senderId,
                PersisterErrorInfo{"The simulation could not be replaced because no valid data could be obtained from the GPU."});
        }

        SerializedSimulation serializedSim;
        if (!SerializerService::get().serializeSimulationToStrings(serializedSim, deserializedSim)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The simulation could not be serialized for replacing."});
        }
        mainData = serializedSim.mainData;
        settings = serializedSim.auxiliaryData;
        statistics = serializedSim.statistics;
        worldSize = {deserializedSim.auxiliaryData.generalSettings.worldSizeX, deserializedSim.auxiliaryData.generalSettings.worldSizeY};
        numObjects = deserializedSim.mainData.getNumberOfCellAndParticles();
    } else {
        auto genome = std::get<ReplaceNetworkResourceRequestData::GenomeData>(requestData.data).description;
        if (genome.cells.empty()) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The is no valid genome for replacement selected."});
        }
        auto genomeData = GenomeDescriptionService::get().convertDescriptionToBytes(genome);
        numObjects = GenomeDescriptionService::get().getNumNodesRecursively(genomeData, true);

        if (!SerializerService::get().serializeGenomeToString(mainData, genomeData)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"The genome could not be serialized for uploading."});
        }
    }

    if (!NetworkService::get().replaceResource(requestData.resourceId, worldSize, numObjects, mainData, settings, statistics)) {

        std::string dataTypeString = resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{
                "Failed to replace " + dataTypeString
                + ".\n\nPossible reasons:\n\n" ICON_FA_CHEVRON_RIGHT " The server is not reachable.\n\n" ICON_FA_CHEVRON_RIGHT
                  " The total size of your uploads exceeds the allowed storage limit."});
    }
    if (resourceType == NetworkResourceType_Simulation) {
        requestData.downloadCache->insertOrAssign(requestData.resourceId, deserializedSim);
    }
    return std::make_shared<_ReplaceNetworkResourceRequestResult>(request->getRequestId(), ReplaceNetworkResourceResultData{});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, GetUserNamesForEmojiRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    GetUserNamesForReactionResultData resultData;
    resultData.resourceId = requestData.resourceId;
    resultData.emojiType = requestData.emojiType;
    if (!NetworkService::get().getUserNamesForResourceAndEmojiType(resultData.userNames, requestData.resourceId, requestData.emojiType)) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Could not load user names."});
    }

    return std::make_shared<_GetUserNamesForEmojiRequestResult>(request->getRequestId(), resultData);
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(
    std::unique_lock<std::mutex>& lock,
    DeleteNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    for (auto const& entry : requestData.entries) {
        if (!NetworkService::get().deleteResource(entry.resourceId)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Failed to delete item. Please try again later."});
        }
    }

    return std::make_shared<_DeleteNetworkResourceRequestResult>(request->getRequestId(), DeleteNetworkResourceResultData{});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, EditNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    for (auto const& entry : requestData.entries) {
        if (!NetworkService::get().editResource(entry.resourceId, entry.newName, entry.newDescription)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Failed to edit item. Please try again later."});
        }
    }

    return std::make_shared<_EditNetworkResourceRequestResult>(request->getRequestId(), EditNetworkResourceResultData{});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, MoveNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    for (auto const& entry : requestData.entries) {
        if (!NetworkService::get().moveResource(entry.resourceId, entry.workspaceType)) {
            return std::make_shared<_PersisterRequestError>(
                request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Failed to change visibility of item. Please try again later."});
        }
    }

    return std::make_shared<_MoveNetworkResourceRequestResult>(request->getRequestId(), MoveNetworkResourceResultData{});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(
    std::unique_lock<std::mutex>& lock,
    ToggleReactionNetworkResourceRequest const& request)
{
    UnlockGuard unlockGuard(lock);

    auto const& requestData = request->getData();

    if (!NetworkService::get().toggleReactionForResource(requestData.resourceId, requestData.emojiType)) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"Failed to toggle reaction. Please try again later."});
    }

    return std::make_shared<_ToggleReactionNetworkResourceRequestResult>(request->getRequestId(), ToggleReactionNetworkResourceResultData{});
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(std::unique_lock<std::mutex>& lock, GetPeakSimulationRequest const& request)
{
    try {
        UnlockGuard unlockGuard(lock);

        auto const& requestData = request->getData();

        auto peakStatistics = requestData.peakDeserializedSimulation->getRawStatisticsData();

        DeserializedSimulation deserializedSimulation;
        deserializedSimulation.statistics = _simulationFacade->getStatisticsHistory().getCopiedData();
        auto currentRawStatistics = _simulationFacade->getRawStatistics();
        if (sumColorVector(currentRawStatistics.timeline.timestep.genomeComplexityVariance)
            >= sumColorVector(peakStatistics.timeline.timestep.genomeComplexityVariance)) {

            deserializedSimulation.auxiliaryData.realTime = _simulationFacade->getRealTime();
            deserializedSimulation.auxiliaryData.zoom = requestData.zoom;
            deserializedSimulation.auxiliaryData.center = requestData.center;
            deserializedSimulation.auxiliaryData.generalSettings = _simulationFacade->getGeneralSettings();
            deserializedSimulation.auxiliaryData.simulationParameters = _simulationFacade->getSimulationParameters();
            deserializedSimulation.auxiliaryData.timestep = static_cast<uint32_t>(_simulationFacade->getCurrentTimestep());
            deserializedSimulation.mainData = _simulationFacade->getClusteredSimulationData();
            requestData.peakDeserializedSimulation->setDeserializedSimulation(std::move(deserializedSimulation));
            requestData.peakDeserializedSimulation->setLastStatisticsData(currentRawStatistics);
        }
        return std::make_shared<_GetPeakSimulationRequestResult>(request->getRequestId(), GetPeakSimulationResultData());
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(), request->getSenderInfo().senderId, PersisterErrorInfo{"No valid data could be obtained from the GPU."});
    }
}

_PersisterWorker::PersisterRequestResultOrError _PersisterWorker::processRequest(
    std::unique_lock<std::mutex>& lock,
    SaveDeserializedSimulationRequest const& request)
{
    try {
        UnlockGuard unlockGuard(lock);

        auto const& requestData = request->getData();

        auto deserializedData = requestData.sharedDeserializedSimulation->getDeserializedSimulation();

        auto filename = requestData.filename;
        if (requestData.generateNameFromTimestep) {
            filename = generateFilename(filename, deserializedData.auxiliaryData.timestep);
        }
        if (!SerializerService::get().serializeSimulationToFiles(filename, deserializedData)) {
            throw std::runtime_error("Error");
        }
        auto result = std::make_shared<_SaveDeserializedSimulationRequestResult>(
            request->getRequestId(),
            SaveDeserializedSimulationResultData{
                .filename = filename,
                .projectName = deserializedData.auxiliaryData.simulationParameters.projectName,
                .timestep = deserializedData.auxiliaryData.timestep,
                .timestamp = requestData.sharedDeserializedSimulation->getTimestamp(),
                .rawStatisticsData = requestData.sharedDeserializedSimulation->getRawStatisticsData()});

        if (requestData.resetDeserializedSimulation) {
            requestData.sharedDeserializedSimulation->reset();
        }
        return result;
    } catch (...) {
        return std::make_shared<_PersisterRequestError>(
            request->getRequestId(),
            request->getSenderInfo().senderId,
            PersisterErrorInfo{"The simulation could not be saved because an error occurred when writing the data to the specified file."});
    }
}
