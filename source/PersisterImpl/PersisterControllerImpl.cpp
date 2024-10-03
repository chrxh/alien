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

PersisterRequestId _PersisterControllerImpl::scheduleReadSimulationFromFile(SenderInfo const& senderInfo, ReadSimulationRequestData const& data)
{
    auto requestId = generateNewJobId();
    auto loadFromFileRequest = std::make_shared<_ReadFromFileRequest>(requestId, senderInfo, data);

    _worker->addRequest(loadFromFileRequest);

    return requestId;
}

ReadSimulationResultData _PersisterControllerImpl::fetchReadSimulationData(PersisterRequestId const& id)
{
    auto requestResult = std::dynamic_pointer_cast<_ReadFromFileRequestResult>(_worker->fetchJobResult(id));
    return requestResult->getData();
}

PersisterRequestId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return PersisterRequestId{std::to_string(_latestJobId)};
}
