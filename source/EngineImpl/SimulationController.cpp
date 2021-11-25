#include "SimulationController.h"

#include "EngineInterface/Descriptions.h"

void _SimulationController::initCuda()
{
    _worker.initCuda();
}

void _SimulationController::newSimulation(uint64_t timestep, Settings const& settings, SymbolMap const& symbolMap)
{
    _settings = settings;
    _origSettings = _settings;
    _symbolMap = symbolMap;
    _worker.newSimulation(timestep, settings, _gpuSettings);
    _origGpuSettings = _gpuSettings;

    _thread = new std::thread(&EngineWorker::runThreadLoop, &_worker);

    _isSelectionInvalid = true;
}

void _SimulationController::clear()
{
    _worker.clear();

    _isSelectionInvalid = true;
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
    _isSelectionInvalid = true;
}

void _SimulationController::calcSingleTimestep()
{
    _worker.calcSingleTimestep();
    _isSelectionInvalid = true;
}

void _SimulationController::runSimulation()
{
    _worker.runSimulation();
    _isSelectionInvalid = true;
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
    _isSelectionInvalid = true;
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
    return _settings.simulationParameters;
}

SimulationParameters _SimulationController::getOriginalSimulationParameters() const
{
    return _origSettings.simulationParameters;
}

void _SimulationController::setSimulationParameters_async(
    SimulationParameters const& parameters)
{
    _settings.simulationParameters = parameters;
    _worker.setSimulationParameters_async(parameters);
}

SimulationParametersSpots _SimulationController::getSimulationParametersSpots() const
{
    return _settings.simulationParametersSpots;
}

SimulationParametersSpots _SimulationController::getOriginalSimulationParametersSpots() const
{
    return _origSettings.simulationParametersSpots;
}

void _SimulationController::setOriginalSimulationParametersSpot(SimulationParametersSpot const& value, int index)
{
    _origSettings.simulationParametersSpots.spots[index] = value;
}

void _SimulationController::setSimulationParametersSpots_async(SimulationParametersSpots const& value)
{
    _settings.simulationParametersSpots = value;
    _worker.setSimulationParametersSpots_async(value);
}

GpuSettings _SimulationController::getGpuSettings() const
{
    return _gpuSettings;
}

GpuSettings _SimulationController::getOriginalGpuSettings() const
{
    return _origGpuSettings;
}

void _SimulationController::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    _gpuSettings = gpuSettings;
    _worker.setGpuSettings_async(gpuSettings);
}

FlowFieldSettings _SimulationController::getFlowFieldSettings() const
{
    return _settings.flowFieldSettings;
}

FlowFieldSettings _SimulationController::getOriginalFlowFieldSettings() const
{
    return _origSettings.flowFieldSettings;
}

void _SimulationController::setOriginalFlowFieldCenter(FlowCenter const& value, int index)
{
    _origSettings.flowFieldSettings.centers[index] = value;
}

void _SimulationController::setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings)
{
    _settings.flowFieldSettings = flowFieldSettings;
    _worker.setFlowFieldSettings_async(flowFieldSettings);
}

void _SimulationController::applyForce_async(
    RealVector2D const& start,
    RealVector2D const& end,
    RealVector2D const& force,
    float radius)
{
    _worker.applyForce_async(start, end, force, radius);
}

void _SimulationController::switchSelection(RealVector2D const& pos, float radius)
{
    _worker.switchSelection(pos, radius);
}

void _SimulationController::getSelection(int& numCells, int& numIndirectCells, int& numParticles)
{
    _worker.getSelection(numCells, numIndirectCells, numParticles);
}

void _SimulationController::setSelection(RealVector2D const& startPos, RealVector2D const& endPos)
{
    _worker.setSelection(startPos, endPos);
}

void _SimulationController::moveSelection(RealVector2D const& displacement)
{
    _worker.moveSelection(displacement);
}

void _SimulationController::removeSelection()
{
    _worker.removeSelection();
}

bool _SimulationController::removeSelectionIfInvalid()
{
    auto result = _isSelectionInvalid;
    _isSelectionInvalid = false;
    if (result) {
        removeSelection();
    }
    return result;
}

GeneralSettings _SimulationController::getGeneralSettings() const
{
    return _settings.generalSettings;
}

IntVector2D _SimulationController::getWorldSize() const
{
    return {_settings.generalSettings.worldSizeX, _settings.generalSettings.worldSizeY};
}

Settings _SimulationController::getSettings() const
{
    return _settings;
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
