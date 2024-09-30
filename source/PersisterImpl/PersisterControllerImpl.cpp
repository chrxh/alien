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

PersisterJobState _PersisterControllerImpl::getJobState(PersisterJobId const& id) const
{
    return _worker->getJobState(id);
}

std::vector<PersisterErrorInfo> _PersisterControllerImpl::fetchCriticalErrorInfos()
{
    return _worker->fetchCriticalErrorInfos();
}

PersisterJobId _PersisterControllerImpl::scheduleSaveSimulationToDisc(std::string const& filename, bool critical, float const& zoom, RealVector2D const& center)
{
    auto jobId = generateNewJobId();
    auto saveToDiscJob = std::make_shared<_SaveToFileJob>(jobId, critical, filename, zoom, center);

    _worker->addJob(saveToDiscJob);

    return jobId;
}

auto _PersisterControllerImpl::fetchSavedSimulationData(PersisterJobId const& id) -> SavedSimulationData
{
    auto saveToDiscResult = std::dynamic_pointer_cast<_SaveToFileJobResult>(_worker->fetchJobResult(id));
    return SavedSimulationData{.name = saveToDiscResult->getSimulationName(), .timestep = saveToDiscResult->getTimestep(), .timestamp = saveToDiscResult->getTimestamp()};
}

PersisterErrorInfo _PersisterControllerImpl::fetchError(PersisterJobId const& id)
{
    return _worker->fetchJobError(id)->getErrorInfo();
}

PersisterJobId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return std::to_string(_latestJobId);
}
