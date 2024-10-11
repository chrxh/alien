#include "PersisterControllerImpl.h"

#include "EngineInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationController.h"

#include "PersisterRequestResult.h"

_PersisterControllerImpl::~_PersisterControllerImpl()
{
    shutdown();
}

void _PersisterControllerImpl::init(SimulationController const& simController)
{
    _worker = std::make_shared<_PersisterWorker>(simController);
    restart();
}

void _PersisterControllerImpl::shutdown()
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

void _PersisterControllerImpl::restart()
{
    _worker->restart();
    for (int i = 0; i < MaxWorkerThreads; ++i) {
        _thread[i] = new std::thread(&_PersisterWorker::runThreadLoop, _worker.get());
    }
}

bool _PersisterControllerImpl::isBusy() const
{
    return _worker->isBusy();
}

PersisterRequestState _PersisterControllerImpl::getRequestState(PersisterRequestId const& id) const
{
    return _worker->getJobState(id);
}

std::vector<PersisterErrorInfo> _PersisterControllerImpl::fetchAllErrorInfos(SenderId const& senderId)
{
    return _worker->fetchAllErrorInfos(senderId);
}

PersisterErrorInfo _PersisterControllerImpl::fetchError(PersisterRequestId const& id)
{
    return _worker->fetchJobError(id)->getErrorInfo();
}

PersisterRequestId _PersisterControllerImpl::scheduleSaveSimulationToFile(SenderInfo const& senderInfo, SaveSimulationRequestData const& data)
{
    return scheduleRequest<_SaveToFileRequest>(senderInfo, data);
}

SavedSimulationResultData _PersisterControllerImpl::fetchSavedSimulationData(PersisterRequestId const& id)
{
    return fetchData<_SaveToFileRequestResult, SavedSimulationResultData>(id);
}

PersisterRequestId _PersisterControllerImpl::scheduleReadSimulationFromFile(SenderInfo const& senderInfo, ReadSimulationRequestData const& data)
{
    return scheduleRequest<_ReadFromFileRequest>(senderInfo, data);
}

ReadSimulationResultData _PersisterControllerImpl::fetchReadSimulationData(PersisterRequestId const& id)
{
    return fetchData<_ReadFromFileRequestResult, ReadSimulationResultData>(id);
}

PersisterRequestId _PersisterControllerImpl::scheduleLogin(SenderInfo const& senderInfo, LoginRequestData const& data)
{
    return scheduleRequest<_LoginRequest>(senderInfo, data);
}

LoginResultData _PersisterControllerImpl::fetchLoginData(PersisterRequestId const& id)
{
    return fetchData<_LoginRequestResult, LoginResultData>(id);
}

PersisterRequestId _PersisterControllerImpl::scheduleGetNetworkResources(SenderInfo const& senderInfo, GetNetworkResourcesRequestData const& data)
{
    return scheduleRequest<_GetNetworkResourcesRequest>(senderInfo, data);
}

GetNetworkResourcesResultData _PersisterControllerImpl::fetchGetNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_GetNetworkResourcesRequestResult, GetNetworkResourcesResultData>(id);
}

PersisterRequestId _PersisterControllerImpl::scheduleDownloadNetworkResource(SenderInfo const& senderInfo, DownloadNetworkResourceRequestData const& data)
{
    return scheduleRequest<_DownloadNetworkResourceRequest>(senderInfo, data);
}

DownloadNetworkResourceResultData _PersisterControllerImpl::fetchDownloadNetworkResourcesData(PersisterRequestId const& id)
{
    return fetchData<_DownloadNetworkResourceRequestResult, DownloadNetworkResourceResultData>(id);
}

PersisterRequestId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return PersisterRequestId{std::to_string(_latestJobId)};
}
