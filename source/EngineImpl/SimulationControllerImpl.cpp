#include "SimulationControllerImpl.h"

#include "EngineInterface/Descriptions.h"

void _SimulationControllerImpl::newSimulation(uint64_t timestep, GeneralSettings const& generalSettings, SimulationParameters const& parameters)
{
    _generalSettings = generalSettings;
    _origSettings.generalSettings = generalSettings;
    _origSettings.simulationParameters = parameters;
    _worker.newSimulation(timestep, generalSettings, parameters);

    _thread = new std::thread(&EngineWorker::runThreadLoop, &_worker);

    _selectionNeedsUpdate = true;
    ++_sessionId;
}

int _SimulationControllerImpl::getSessionId() const
{
    return _sessionId;
}

void _SimulationControllerImpl::clear()
{
    _worker.clear();

    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::setImageResource(void* image)
{
    _worker.setImageResource(image);
}

std::string _SimulationControllerImpl::getGpuName() const
{
    return _worker.getGpuName();
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

bool _SimulationControllerImpl::isSyncSimulationWithRendering() const
{
    return _worker.isSyncSimulationWithRendering();
}

void _SimulationControllerImpl::setSyncSimulationWithRendering(bool value)
{
    _worker.setSyncSimulationWithRendering(value);
}

int _SimulationControllerImpl::getSyncSimulationWithRenderingRatio() const
{
    return _worker.getSyncSimulationWithRenderingRatio();
}

void _SimulationControllerImpl::setSyncSimulationWithRenderingRatio(int value)
{
    _worker.setSyncSimulationWithRenderingRatio(value);
}

ClusteredDataDescription _SimulationControllerImpl::getClusteredSimulationData()
{
    auto size = getWorldSize();
    return _worker.getClusteredSimulationData({-10, -10}, {size.x + 10, size.y + 10});
}

DataDescription _SimulationControllerImpl::getSimulationData()
{
    auto size = getWorldSize();
    return _worker.getSimulationData({-10, -10}, {size.x + 10, size.y + 10});
}

ClusteredDataDescription _SimulationControllerImpl::getSelectedClusteredSimulationData(bool includeClusters)
{
    _worker.updateSelection();
    return _worker.getSelectedClusteredSimulationData(includeClusters);
}

DataDescription _SimulationControllerImpl::getSelectedSimulationData(bool includeClusters)
{
    _worker.updateSelection();
    return _worker.getSelectedSimulationData(includeClusters);
}

DataDescription _SimulationControllerImpl::getInspectedSimulationData(std::vector<uint64_t> objectIds)
{
    return _worker.getInspectedSimulationData(objectIds);
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

void _SimulationControllerImpl::removeSelectedObjects(bool includeClusters)
{
    _worker.removeSelectedObjects(includeClusters);
    _selectionNeedsUpdate = true;
}

void _SimulationControllerImpl::relaxSelectedObjects(bool includeClusters)
{
    _worker.relaxSelectedObjects(includeClusters);
}

void _SimulationControllerImpl::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    _worker.uniformVelocitiesForSelectedObjects(includeClusters);
}

void _SimulationControllerImpl::makeSticky(bool includeClusters)
{
    _worker.makeSticky(includeClusters);
}

void _SimulationControllerImpl::removeStickiness(bool includeClusters)
{
    _worker.removeStickiness(includeClusters);
}

void _SimulationControllerImpl::setBarrier(bool value, bool includeClusters)
{
    _worker.setBarrier(value, includeClusters);
}

void _SimulationControllerImpl::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    _worker.colorSelectedObjects(color, includeClusters);
}

void _SimulationControllerImpl::reconnectSelectedObjects()
{
    _worker.reconnectSelectedObjects();
}

void _SimulationControllerImpl::setDetached(bool value)
{
    _worker.setDetached(value);
}

void _SimulationControllerImpl::changeCell(CellDescription const& changedCell)
{
    _worker.changeCell(changedCell);
}

void _SimulationControllerImpl::changeParticle(ParticleDescription const& changedParticle)
{
    _worker.changeParticle(changedParticle);
}

void _SimulationControllerImpl::calcTimesteps(uint64_t timesteps)
{
    _worker.calcTimesteps(timesteps);
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

void _SimulationControllerImpl::applyCataclysm(int power)
{
    _worker.applyCataclysm(power);
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

SimulationParameters _SimulationControllerImpl::getSimulationParameters() const
{
    return _worker.getSimulationParameters();
}

SimulationParameters const& _SimulationControllerImpl::getOriginalSimulationParameters() const
{
    return _origSettings.simulationParameters;
}

void _SimulationControllerImpl::setSimulationParameters(SimulationParameters const& parameters)
{
    _worker.setSimulationParameters(parameters);
}

void _SimulationControllerImpl::setOriginalSimulationParameters(SimulationParameters const& parameters)
{
    _origSettings.simulationParameters = parameters;
}

GpuSettings _SimulationControllerImpl::getGpuSettings() const
{
    return _gpuSettings;
}

GpuSettings _SimulationControllerImpl::getOriginalGpuSettings() const
{
    return _origSettings.gpuSettings;
}

void _SimulationControllerImpl::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    _gpuSettings = gpuSettings;
    _worker.setGpuSettings_async(gpuSettings);
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

void _SimulationControllerImpl::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData)
{
    _worker.shallowUpdateSelectedObjects(updateData);
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
    return _generalSettings;
}

IntVector2D _SimulationControllerImpl::getWorldSize() const
{
    return {_generalSettings.worldSizeX, _generalSettings.worldSizeY};
}

RawStatisticsData _SimulationControllerImpl::getRawStatistics() const
{
    return _worker.getRawStatistics();
}

StatisticsHistory const& _SimulationControllerImpl::getStatisticsHistory() const
{
    return _worker.getStatisticsHistory();
}

void _SimulationControllerImpl::setStatisticsHistory(StatisticsHistoryData const& data)
{
    _worker.setStatisticsHistory(data);
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

void _SimulationControllerImpl::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    _worker.testOnly_mutate(cellId, mutationType);
}