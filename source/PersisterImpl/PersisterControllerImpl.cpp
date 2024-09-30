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

PersisterErrorInfo _PersisterControllerImpl::fetchErrorInfo() const
{
    return PersisterErrorInfo();
    //_worker->fetchErrorInfo();
}

PersisterJobId _PersisterControllerImpl::scheduleSaveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center)
{
    auto jobId = generateNewJobId();
    auto saveToDiscJob = std::make_shared<_SaveToDiscJob>(jobId, filename, zoom, center);

    _worker->addJob(saveToDiscJob);

    return jobId;
}

_PersisterController::SavedSimulationData _PersisterControllerImpl::fetchSavedSimulationData(PersisterJobId const& id)
{
    auto jobResult = std::dynamic_pointer_cast<_SaveToDiscJobResult>(_worker->fetchJobResult(id));
    return {.name = "", .timestep = jobResult->getTimestep(), .realtime = jobResult->getRealtime()};
}

PersisterJobId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return std::to_string(_latestJobId);
}
