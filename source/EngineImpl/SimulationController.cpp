#include "SimulationController.h"

namespace
{
    void runEngineWorker(EngineWorker* worker)
    {
    }
}

void _SimulationController::initCuda()
{
    _worker.initCuda();
}

void _SimulationController::newSimulation(IntVector2D size, int timestep, SimulationParameters const& parameters, GpuConstants const& gpuConstants)
{
    _worldSize = size;
    _worker.newSimulation(size, timestep, parameters, gpuConstants);

    _thread = new std::thread(runEngineWorker, &_worker);

}

void* _SimulationController::registerImageResource(GLuint image)
{
    return _worker.registerImageResource(image);
}

void _SimulationController::getVectorImage(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    void* const& resource,
    IntVector2D const& imageSize,
    double zoom)
{
    _worker.getVectorImage(rectUpperLeft, rectLowerRight, resource, imageSize, zoom);
}

void _SimulationController::updateData(DataChangeDescription const& dataToUpdate)
{
    _worker.updateData(dataToUpdate);
}

void _SimulationController::calcNextTimestep()
{
    _worker.calcNextTimestep();
}

void _SimulationController::closeSimulation()
{
    _thread->join();
    delete _thread;

    _worker.shutdown();
}

IntVector2D _SimulationController::getWorldSize() const
{
    return _worldSize;
}
