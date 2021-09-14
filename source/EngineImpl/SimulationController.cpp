#include "SimulationController.h"

#include "EngineInterface/Descriptions.h"

void _SimulationController::initCuda()
{
    _worker.initCuda();
}

void _SimulationController::newSimulation(IntVector2D size, int timestep, SimulationParameters const& parameters, GpuConstants const& gpuConstants)
{
    _worldSize = size;
    _worker.newSimulation(size, timestep, parameters, gpuConstants);

    _thread = new std::thread(&EngineWorker::runThreadLoop, &_worker);
}

void _SimulationController::clear()
{
    _worker.clear();
}

void _SimulationController::registerImageResource(GLuint image)
{
    _worker.registerImageResource(image);
}

void _SimulationController::getVectorImage(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    _worker.getVectorImage(rectUpperLeft, rectLowerRight, imageSize, zoom);
}

DataDescription _SimulationController::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    return _worker.getSimulationData(rectUpperLeft, rectLowerRight);
}

void _SimulationController::updateData(DataChangeDescription const& dataToUpdate)
{
    _worker.updateData(dataToUpdate);
}

void _SimulationController::calcSingleTimestep()
{
    _worker.calcSingleTimestep();
}

void _SimulationController::runSimulation()
{
    _worker.runSimulation();
}

void _SimulationController::pauseSimulation()
{
    _worker.pauseSimulation();
}

bool _SimulationController::isSimulationRunning() const
{
    return _worker.isSimulationRunning();
}

void _SimulationController::closeSimulation()
{
    _worker.beginShutdown();
    _thread->join();
    delete _thread;
    _worker.endShutdown();
}

uint64_t _SimulationController::getCurrentTimestep() const
{
    return _worker.getCurrentTimestep();
}

void _SimulationController::setCurrentTimestep(uint64_t value)
{
    _worker.setCurrentTimestep(value);
}

IntVector2D _SimulationController::getWorldSize() const
{
    return _worldSize;
}

boost::optional<int> _SimulationController::getTpsRestriction() const
{
    auto result = _worker.getTpsRestriction();
    return 0 != result ? boost::optional<int>(result) : boost::optional<int>();
}

void _SimulationController::setTpsRestriction(boost::optional<int> const& value)
{
    _worker.setTpsRestriction(value ? *value : 0);
}

int _SimulationController::getTps() const
{
    return _worker.getTps();
}
