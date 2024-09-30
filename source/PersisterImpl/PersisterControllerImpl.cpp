#include "PersisterControllerImpl.h"

#include "EngineInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationController.h"

#include "PersisterWorker.h"
#include "PersisterRequestResult.h"

_PersisterControllerImpl::~_PersisterControllerImpl()
{
    shutdown();
}

void _PersisterControllerImpl::init(SimulationController const& simController)
{
    _worker = std::make_shared<_PersisterWorker>(simController);
    for (int i = 0; i < MaxWorkerThreads; ++i) {
        _thread[i] = new std::thread(&_PersisterWorker::runThreadLoop, _worker.get());
    }
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
    auto requestId = generateNewJobId();
    auto saveToFileRequest = std::make_shared<_SaveToFileRequest>(requestId, senderInfo, data);

    _worker->addRequest(saveToFileRequest);

    return requestId;
}

SavedSimulationResultData _PersisterControllerImpl::fetchSavedSimulationData(PersisterRequestId const& id)
{
    auto requestResult = std::dynamic_pointer_cast<_SaveToFileRequestResult>(_worker->fetchJobResult(id));
    return requestResult->getData();
}

PersisterRequestId _PersisterControllerImpl::scheduleLoadSimulationFromFile(SenderInfo const& senderInfo, LoadSimulationRequestData const& data)
{
    auto requestId = generateNewJobId();
    auto loadFromFileRequest = std::make_shared<_LoadFromFileRequest>(requestId, senderInfo, data);

    _worker->addRequest(loadFromFileRequest);

    return requestId;
}

LoadedSimulationResultData _PersisterControllerImpl::fetchLoadSimulationData(PersisterRequestId const& id)
{
    auto requestResult = std::dynamic_pointer_cast<_LoadFromFileRequestResult>(_worker->fetchJobResult(id));
    return requestResult->getData();
}

PersisterRequestId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return PersisterRequestId{std::to_string(_latestJobId)};
}
