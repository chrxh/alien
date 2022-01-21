#include "SimulationControllerImpl.h"

#include "EngineInterface/Descriptions.h"

void _SimulationControllerImpl::initCuda()
{
    _worker.initCuda();
}

void _SimulationControllerImpl::newSimulation(uint64_t timestep, Settings const& settings, SymbolMap const& symbolMap)
{
    _settings = settings;
    _origSettings = _settings;
    _symbolMap = symbolMap;
    _worker.newSimulation(timestep, settings, _gpuSettings);
    _origGpuSettings = _gpuSettings;

    _thread = new std::thread(&EngineWorker::runThreadLoop, &_worker);

    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::clear()
{
    _worker.clear();

    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::registerImageResource(void* image)
{
    _worker.registerImageResource(image);
}

void _SimulationControllerImpl::tryDrawVectorGraphics(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    _worker.tryDrawVectorGraphics(rectUpperLeft, rectLowerRight, imageSize, zoom);
}

std::optional<OverlayDescription> _SimulationControllerImpl::tryDrawVectorGraphicsAndReturnOverlay(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    return _worker.tryDrawVectorGraphicsAndReturnOverlay(rectUpperLeft, rectLowerRight, imageSize, zoom);
}

ClusteredDataDescription _SimulationControllerImpl::getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    return _worker.getClusteredSimulationData(rectUpperLeft, rectLowerRight);
}

DataDescription _SimulationControllerImpl::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    return _worker.getSimulationData(rectUpperLeft, rectLowerRight);
}

ClusteredDataDescription _SimulationControllerImpl::getSelectedClusteredSimulationData(bool includeClusters)
{
    return _worker.getSelectedClusteredSimulationData(includeClusters);
}

DataDescription _SimulationControllerImpl::getSelectedSimulationData(bool includeClusters)
{
    return _worker.getSelectedSimulationData(includeClusters);
}

DataDescription _SimulationControllerImpl::getInspectedSimulationData(std::vector<uint64_t> entityIds)
{
    return _worker.getInspectedSimulationData(entityIds);
}

void _SimulationControllerImpl::addAndSelectSimulationData(DataDescription const& dataToAdd)
{
    _worker.addAndSelectSimulationData(dataToAdd);
}

void _SimulationControllerImpl::setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate)
{
    _worker.setClusteredSimulationData(dataToUpdate);
    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::setSimulationData(DataDescription const& dataToUpdate)
{
    _worker.setSimulationData(dataToUpdate);
    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::removeSelectedEntities(bool includeClusters)
{
    _worker.removeSelectedEntities(includeClusters);
    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::colorSelectedEntities(unsigned char color, bool includeClusters)
{
    _worker.colorSelectedEntities(color, includeClusters);
}

void _SimulationControllerImpl::reconnectSelectedEntities()
{
    _worker.reconnectSelectedEntities();
}

void _SimulationControllerImpl::changeCell(CellDescription const& changedCell)
{
    _worker.changeCell(changedCell);
}

void _SimulationControllerImpl::changeParticle(ParticleDescription const& changedParticle)
{
    _worker.changeParticle(changedParticle);
}

void _SimulationControllerImpl::calcSingleTimestep()
{
    _worker.calcSingleTimestep();
    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::runSimulation()
{
    _worker.runSimulation();
}

void _SimulationControllerImpl::pauseSimulation()
{
    _worker.pauseSimulation();
    _selectionNeedsUpdate = true;
}

bool _SimulationControllerImpl::isSimulationRunning() const
{
    return _worker.isSimulationRunning();
}

void _SimulationControllerImpl::closeSimulation()
{
    _worker.beginShutdown();
    _thread->join();
    delete _thread;
    _worker.endShutdown();
    _selectionNeedsUpdate = true;
}

uint64_t _SimulationControllerImpl::getCurrentTimestep() const
{
    return _worker.getCurrentTimestep();
}

void _SimulationControllerImpl::setCurrentTimestep(uint64_t value)
{
    _worker.setCurrentTimestep(value);
}

SimulationParameters const& _SimulationControllerImpl::getSimulationParameters() const
{
    return _settings.simulationParameters;
}

SimulationParameters _SimulationControllerImpl::getOriginalSimulationParameters() const
{
    return _origSettings.simulationParameters;
}

void _SimulationControllerImpl::setSimulationParameters_async(
    SimulationParameters const& parameters)
{
    _settings.simulationParameters = parameters;
    _worker.setSimulationParameters_async(parameters);
}

SimulationParametersSpots _SimulationControllerImpl::getSimulationParametersSpots() const
{
    return _settings.simulationParametersSpots;
}

SimulationParametersSpots _SimulationControllerImpl::getOriginalSimulationParametersSpots() const
{
    return _origSettings.simulationParametersSpots;
}

void _SimulationControllerImpl::setOriginalSimulationParametersSpot(SimulationParametersSpot const& value, int index)
{
    _origSettings.simulationParametersSpots.spots[index] = value;
}

void _SimulationControllerImpl::setSimulationParametersSpots_async(SimulationParametersSpots const& value)
{
    _settings.simulationParametersSpots = value;
    _worker.setSimulationParametersSpots_async(value);
}

GpuSettings _SimulationControllerImpl::getGpuSettings() const
{
    return _gpuSettings;
}

GpuSettings _SimulationControllerImpl::getOriginalGpuSettings() const
{
    return _origGpuSettings;
}

void _SimulationControllerImpl::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    _gpuSettings = gpuSettings;
    _worker.setGpuSettings_async(gpuSettings);
}

FlowFieldSettings _SimulationControllerImpl::getFlowFieldSettings() const
{
    return _settings.flowFieldSettings;
}

FlowFieldSettings _SimulationControllerImpl::getOriginalFlowFieldSettings() const
{
    return _origSettings.flowFieldSettings;
}

void _SimulationControllerImpl::setOriginalFlowFieldCenter(FlowCenter const& value, int index)
{
    _origSettings.flowFieldSettings.centers[index] = value;
}

void _SimulationControllerImpl::setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings)
{
    _settings.flowFieldSettings = flowFieldSettings;
    _worker.setFlowFieldSettings_async(flowFieldSettings);
}

void _SimulationControllerImpl::applyForce_async(
    RealVector2D const& start,
    RealVector2D const& end,
    RealVector2D const& force,
    float radius)
{
    _worker.applyForce_async(start, end, force, radius);
}

void _SimulationControllerImpl::switchSelection(RealVector2D const& pos, float radius)
{
    _worker.switchSelection(pos, radius);
}

void _SimulationControllerImpl::swapSelection(RealVector2D const& pos, float radius)
{
    _worker.swapSelection(pos, radius);
}

SelectionShallowData _SimulationControllerImpl::getSelectionShallowData()
{
    return _worker.getSelectionShallowData();
}

void _SimulationControllerImpl::shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& updateData)
{
    _worker.shallowUpdateSelectedEntities(updateData);
}

void _SimulationControllerImpl::setSelection(RealVector2D const& startPos, RealVector2D const& endPos)
{
    _worker.setSelection(startPos, endPos);
}

void _SimulationControllerImpl::removeSelection()
{
    _worker.removeSelection();
}

bool _SimulationControllerImpl::updateSelectionIfNecessary()
{
    auto result = _selectionNeedsUpdate;
    _selectionNeedsUpdate = false;
    if (result) {
        _worker.updateSelection();
    }
    return result;
}

GeneralSettings _SimulationControllerImpl::getGeneralSettings() const
{
    return _settings.generalSettings;
}

IntVector2D _SimulationControllerImpl::getWorldSize() const
{
    return {_settings.generalSettings.worldSizeX, _settings.generalSettings.worldSizeY};
}

Settings _SimulationControllerImpl::getSettings() const
{
    return _settings;
}

SymbolMap const& _SimulationControllerImpl::getSymbolMap() const
{
    return _symbolMap;
}

void _SimulationControllerImpl::setSymbolMap(SymbolMap const& symbolMap)
{
    _symbolMap = symbolMap;
}

OverallStatistics _SimulationControllerImpl::getStatistics() const
{
    return _worker.getMonitorData();
}

std::optional<int> _SimulationControllerImpl::getTpsRestriction() const
{
    auto result = _worker.getTpsRestriction();
    return 0 != result ? std::optional<int>(result) : std::optional<int>();
}

void _SimulationControllerImpl::setTpsRestriction(std::optional<int> const& value)
{
    _worker.setTpsRestriction(value ? *value : 0);
}

float _SimulationControllerImpl::getTps() const
{
    return _worker.getTps();
}
