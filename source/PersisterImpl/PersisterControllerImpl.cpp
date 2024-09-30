#include "PersisterControllerImpl.h"

#include "EngineInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationController.h"

#include "PersisterWorker.h"

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

PersisterRequestState _PersisterControllerImpl::getJobState(PersisterRequestId const& id) const
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

PersisterRequestId _PersisterControllerImpl::scheduleSaveSimulationToFile(
    SenderInfo const& senderInfo,
    std::string const& filename,
    float const& zoom,
    RealVector2D const& center)
{
    auto requestId = generateNewJobId();
    auto saveToFileJob = std::make_shared<_SaveToFileJob>(requestId, senderInfo, filename, zoom, center);

    _worker->addJob(saveToFileJob);

    return requestId;
}

auto _PersisterControllerImpl::fetchSavedSimulationData(PersisterRequestId const& id) -> SavedSimulationData
{
    auto saveToDiscResult = std::dynamic_pointer_cast<_SaveToFileJobResult>(_worker->fetchJobResult(id));
    return SavedSimulationData{.name = saveToDiscResult->getSimulationName(), .timestep = saveToDiscResult->getTimestep(), .timestamp = saveToDiscResult->getTimestamp()};
}

PersisterRequestId _PersisterControllerImpl::scheduleLoadSimulationFromFile(SenderInfo const& senderInfo, std::string const& filename)
{
    auto requestId = generateNewJobId();
    auto loadFromFileJob = std::make_shared<_LoadFromFileJob>(requestId, senderInfo, filename);

    _worker->addJob(loadFromFileJob);

    return requestId;
}

PersisterRequestId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return PersisterRequestId{std::to_string(_latestJobId)};
}
