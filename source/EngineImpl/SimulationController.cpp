#include "SimulationController.h"

#include "EngineInterface/Descriptions.h"

void _SimulationController::initCuda()
{
    _worker.initCuda();
}

void _SimulationController::newSimulation(
    int timestep,
    GeneralSettings const& generalSettings,
    SimulationParameters const& parameters,
    SymbolMap const& symbolMap)
{
    _generalSettings = generalSettings;
    _parameters = parameters;
    _symbolMap = symbolMap;
    _worker.newSimulation(generalSettings.worldSize, timestep, parameters, _gpuSettings);

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

SimulationParameters _SimulationController::getSimulationParameters() const
{
    return _parameters;
}

void _SimulationController::setSimulationParameters_async(
    SimulationParameters const& parameters)
{
    _parameters = parameters;
    _worker.setSimulationParameters_async(parameters);
}

GpuConstants _SimulationController::getGpuSettings() const
{
    return _gpuSettings;
}

void _SimulationController::setGpuSettings_async(GpuConstants const& gpuSettings)
{
    _gpuSettings = gpuSettings;
    _worker.setGpuSettings_async(gpuSettings);
}

void _SimulationController::applyForce_async(
    RealVector2D const& start,
    RealVector2D const& end,
    RealVector2D const& force,
    float radius)
{
    _worker.applyForce_async(start, end, force, radius);
}

GeneralSettings _SimulationController::getGeneralSettings() const
{
    return _generalSettings;
}

IntVector2D _SimulationController::getWorldSize() const
{
    return _generalSettings.worldSize;
}

SymbolMap _SimulationController::getSymbolMap() const
{
    return _symbolMap;
}

OverallStatistics _SimulationController::getStatistics() const
{
    return _worker.getMonitorData();
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

float _SimulationController::getTps() const
{
    return _worker.getTps();
}
