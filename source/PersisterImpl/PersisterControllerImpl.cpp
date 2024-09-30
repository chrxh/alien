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

void _PersisterControllerImpl::saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center)
{
    DeserializedSimulation dummy;
    _worker->saveSimulationToDisc("d:\\test.sim", zoom, center);
}
