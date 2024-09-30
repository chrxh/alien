#include "PersisterControllerImpl.h"

void _PersisterControllerImpl::init(SimulationController const& simController)
{
    _thread = new std::thread(&PersisterWorker::runThreadLoop, &_worker);
}

void _PersisterControllerImpl::shutdown()
{
    _worker.shutdown();
    _thread->join();
    delete _thread;
}

void _PersisterControllerImpl::saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center)
{
    DeserializedSimulation dummy;
    _worker.saveSimulationToDisc("d:\\test.sim", zoom, center);
}
