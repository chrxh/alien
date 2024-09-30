#include "PersisterControllerImpl.h"

#include "EngineInterface/DeserializedSimulation.h"
#include "EngineInterface/SimulationController.h"

#include "PersisterWorker.h"

_PersisterControllerImpl::~_PersisterControllerImpl()
{
    if (_thread) {
        shutdown();
    }
}

void _PersisterControllerImpl::init(SimulationController const& simController)
{
    _worker = std::make_shared<_PersisterWorker>(simController);
    _thread = new std::thread(&_PersisterWorker::runThreadLoop, _worker.get());
}

void _PersisterControllerImpl::shutdown()
{
    _worker->shutdown();
    _thread->join();
    delete _thread;
    _thread = nullptr;
}

PersisterJobState _PersisterControllerImpl::getJobState(PersisterJobId const& id) const
{
    return _worker->getJobState(id);
}

PersisterJobId _PersisterControllerImpl::saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center)
{
    auto jobId = generateNewJobId();
    auto saveToDiscJob = std::make_shared<_SaveToDiscJob>(jobId, filename, zoom, center);

    _worker->addJob(saveToDiscJob);

    return jobId;
}

PersisterJobId _PersisterControllerImpl::generateNewJobId()
{
    ++_latestJobId;
    return _latestJobId;
}
