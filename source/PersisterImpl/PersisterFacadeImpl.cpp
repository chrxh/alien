#include "PersisterFacadeImpl.h"

#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/DeserializedSimulation.h"
#include "PersisterInterface/PersisterRequestResult.h"

_PersisterFacadeImpl::~_PersisterFacadeImpl()
{
    shutdown();
}

void _PersisterFacadeImpl::setup(SimulationFacade const& simulationFacade)
{
    _worker = std::make_shared<_PersisterWorker>(simulationFacade);
    restart();
}

void _PersisterFacadeImpl::shutdown()
{
    _worker->shutdown();
    for (int i = 0; i < MaxWorkerThreads; ++i) {
        if (_thread[i]) {
            _thread[i]->join();
            delete _thread[i];
            _thread[i] = nullptr;
        }
    }
}

void _PersisterFacadeImpl::restart()
{
    _worker->restart();
    for (int i = 0; i < MaxWorkerThreads; ++i) {
        _thread[i] = new std::thread(&_PersisterWorker::runThreadLoop, _worker.get());
    }
}

bool _PersisterFacadeImpl::isBusy() const
{
    return _worker->isBusy();
}

std::optional<PersisterRequestState> _PersisterFacadeImpl::getRequestState(PersisterRequestId const& id) const
{
    return _worker->getRequestState(id);
}

PersisterRequestResult _PersisterFacadeImpl::fetchPersisterRequestResult(PersisterRequestId const& id)
{
    return _worker->fetchRequestResult(id);
}

std::vector<PersisterErrorInfo> _PersisterFacadeImpl::fetchAllErrorInfos(SenderId const& senderId)
{
    return _worker->fetchAllErrorInfos(senderId);
}

PersisterErrorInfo _PersisterFacadeImpl::fetchError(PersisterRequestId const& id)
{
    return _worker->fetchJobError(id)->getErrorInfo();
}

PersisterRequestId _PersisterFacadeImpl::scheduleSaveSimulation(SenderInfo const& senderInfo, SaveSimulationRequestData const& data)
{
    return scheduleRequest<_SaveSimulationRequest>(senderInfo, data);
}

SaveSimulationResultData _PersisterFacadeImpl::fetchSaveSimulationData(PersisterRequestId const& id)
{
    return fetchData<_SaveSimulationRequestResult, SaveSimulationResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleReadSimulation(SenderInfo const& senderInfo, ReadSimulationRequestData const& data)
{
    return scheduleRequest<_ReadSimulationRequest>(senderInfo, data);
}

ReadSimulationResultData _PersisterFacadeImpl::fetchReadSimulationData(PersisterRequestId const& id)
{
    return fetchData<_ReadSimulationRequestResult, ReadSimulationResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleLogin(SenderInfo const& senderInfo, LoginRequestData const& data)
{
    return scheduleRequest<_LoginRequest>(senderInfo, data);
}

LoginResultData _PersisterFacadeImpl::fetchLoginData(PersisterRequestId const& id)
{
    return fetchData<_LoginRequestResult, LoginResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleGetNetworkResources(SenderInfo const& senderInfo, GetNetworkResourcesRequestData const& data)
{
    return scheduleRequest<_GetNetworkResourcesRequest>(senderInfo, data);
}

GetNetworkResourcesResultData _PersisterFacadeImpl::fetchGetNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_GetNetworkResourcesRequestResult, GetNetworkResourcesResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleDownloadNetworkResource(SenderInfo const& senderInfo, DownloadNetworkResourceRequestData const& data)
{
    return scheduleRequest<_DownloadNetworkResourceRequest>(senderInfo, data);
}

DownloadNetworkResourceResultData _PersisterFacadeImpl::fetchDownloadNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_DownloadNetworkResourceRequestResult, DownloadNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleUploadNetworkResource(SenderInfo const& senderInfo, UploadNetworkResourceRequestData const& data)
{
    return scheduleRequest<_UploadNetworkResourceRequest>(senderInfo, data);
}

UploadNetworkResourceResultData _PersisterFacadeImpl::fetchUploadNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_UploadNetworkResourceRequestResult, UploadNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleReplaceNetworkResource(SenderInfo const& senderInfo, ReplaceNetworkResourceRequestData const& data)
{
    return scheduleRequest<_ReplaceNetworkResourceRequest>(senderInfo, data);
}

ReplaceNetworkResourceResultData _PersisterFacadeImpl::fetchReplaceNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_ReplaceNetworkResourceRequestResult, ReplaceNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleGetUserNamesForReaction(SenderInfo const& senderInfo, GetUserNamesForReactionRequestData const& data)
{
    return scheduleRequest<_GetUserNamesForEmojiRequest>(senderInfo, data);
}

GetUserNamesForReactionResultData _PersisterFacadeImpl::fetchGetUserNamesForReactionData(PersisterRequestId const& id)
{
    return fetchData<_GetUserNamesForEmojiRequestResult, GetUserNamesForReactionResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleDeleteNetworkResource(SenderInfo const& senderInfo, DeleteNetworkResourceRequestData const& data)
{
    return scheduleRequest<_DeleteNetworkResourceRequest>(senderInfo, data);
}

DeleteNetworkResourceResultData _PersisterFacadeImpl::fetchDeleteNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_DeleteNetworkResourceRequestResult, DeleteNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleEditNetworkResource(SenderInfo const& senderInfo, EditNetworkResourceRequestData const& data)
{
    return scheduleRequest<_EditNetworkResourceRequest>(senderInfo, data);
}

EditNetworkResourceResultData _PersisterFacadeImpl::fetchEditNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_EditNetworkResourceRequestResult, EditNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleMoveNetworkResource(SenderInfo const& senderInfo, MoveNetworkResourceRequestData const& data)
{
    return scheduleRequest<_MoveNetworkResourceRequest>(senderInfo, data);
}

MoveNetworkResourceResultData _PersisterFacadeImpl::fetchMoveNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_MoveNetworkResourceRequestResult, MoveNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleToggleReactionNetworkResource(SenderInfo const& senderInfo, ToggleReactionNetworkResourceRequestData const& data)
{
    return scheduleRequest<_ToggleReactionNetworkResourceRequest>(senderInfo, data);
}

ToggleReactionNetworkResourceResultData _PersisterFacadeImpl::fetchToggleReactionNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_ToggleReactionNetworkResourceRequestResult, ToggleReactionNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleGetPeakSimulation(SenderInfo const& senderInfo, GetPeakSimulationRequestData const& data)
{
    return scheduleRequest<_GetPeakSimulationRequest>(senderInfo, data);
}

GetPeakSimulationResultData _PersisterFacadeImpl::fetchGetPeakSimulationData(PersisterRequestId const& id)
{
    return fetchData<_GetPeakSimulationRequestResult, GetPeakSimulationResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::scheduleSaveDeserializedSimulation(SenderInfo const& senderInfo, SaveDeserializedSimulationRequestData const& data)
{
    return scheduleRequest<_SaveDeserializedSimulationRequest>(senderInfo, data);
}

SaveDeserializedSimulationResultData _PersisterFacadeImpl::fetchSaveDeserializedSimulationData(PersisterRequestId const& id)
{
    return fetchData<_SaveDeserializedSimulationRequestResult, SaveDeserializedSimulationResultData>(id);
}

PersisterRequestId _PersisterFacadeImpl::generateNewRequestId()
{
    ++_latestRequestId;
    return PersisterRequestId{std::to_string(_latestRequestId)};
}
