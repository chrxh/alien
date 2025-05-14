#include "SimulationFacadeImpl.h"

#include "EngineInterface/Descriptions.h"

void _SimulationFacadeImpl::newSimulation(uint64_t timestep, IntVector2D const& worldSize, SimulationParameters const& parameters)
{
    _worldSize = worldSize;
    _origSettings.worldSizeX = worldSize.x;
    _origSettings.worldSizeY = worldSize.y;
    _origSettings.simulationParameters = parameters;
    _worker.newSimulation(timestep, _origSettings);

    _thread = new std::thread(&EngineWorker::runThreadLoop, &_worker);

    _selectionNeedsUpdate = true;
    _realTime = std::chrono::milliseconds(0);
    _simRunTimePoint.reset();

    ++_sessionId;
}

int _SimulationFacadeImpl::getSessionId() const
{
    return _sessionId;
}

void _SimulationFacadeImpl::clear()
{
    _worker.clear();

    _selectionNeedsUpdate = true;
}

void _SimulationFacadeImpl::setImageResource(void* image)
{
    _worker.setImageResource(image);
}

std::string _SimulationFacadeImpl::getGpuName() const
{
    return _worker.getGpuName();
}

void _SimulationFacadeImpl::tryDrawVectorGraphics(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    _worker.tryDrawVectorGraphics(rectUpperLeft, rectLowerRight, imageSize, zoom);
}

std::optional<OverlayDescription> _SimulationFacadeImpl::tryDrawVectorGraphicsAndReturnOverlay(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    return _worker.tryDrawVectorGraphicsAndReturnOverlay(rectUpperLeft, rectLowerRight, imageSize, zoom);
}

bool _SimulationFacadeImpl::isSyncSimulationWithRendering() const
{
    return _worker.isSyncSimulationWithRendering();
}

void _SimulationFacadeImpl::setSyncSimulationWithRendering(bool value)
{
    _worker.setSyncSimulationWithRendering(value);
}

int _SimulationFacadeImpl::getSyncSimulationWithRenderingRatio() const
{
    return _worker.getSyncSimulationWithRenderingRatio();
}

void _SimulationFacadeImpl::setSyncSimulationWithRenderingRatio(int value)
{
    _worker.setSyncSimulationWithRenderingRatio(value);
}

ClusteredDataDescription _SimulationFacadeImpl::getClusteredSimulationData()
{
    auto size = getWorldSize();
    return _worker.getClusteredSimulationData({-10, -10}, {size.x + 10, size.y + 10});
}

DataDescription _SimulationFacadeImpl::getSimulationData()
{
    auto size = getWorldSize();
    return _worker.getSimulationData({-10, -10}, {size.x + 10, size.y + 10});
}

ClusteredDataDescription _SimulationFacadeImpl::getSelectedClusteredSimulationData(bool includeClusters)
{
    _worker.updateSelection();
    return _worker.getSelectedClusteredSimulationData(includeClusters);
}

DataDescription _SimulationFacadeImpl::getSelectedSimulationData(bool includeClusters)
{
    _worker.updateSelection();
    return _worker.getSelectedSimulationData(includeClusters);
}

DataDescription _SimulationFacadeImpl::getInspectedSimulationData(std::vector<uint64_t> objectIds)
{
    return _worker.getInspectedSimulationData(objectIds);
}

void _SimulationFacadeImpl::addAndSelectSimulationData(DataDescription const& dataToAdd)
{
    _worker.addAndSelectSimulationData(dataToAdd);
}

void _SimulationFacadeImpl::setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate)
{
    _worker.setClusteredSimulationData(dataToUpdate);
    _selectionNeedsUpdate = true;
}

void _SimulationFacadeImpl::setSimulationData(DataDescription const& dataToUpdate)
{
    _worker.setSimulationData(dataToUpdate);
    _selectionNeedsUpdate = true;
}

void _SimulationFacadeImpl::removeSelectedObjects(bool includeClusters)
{
    _worker.removeSelectedObjects(includeClusters);
    _selectionNeedsUpdate = true;
}

void _SimulationFacadeImpl::relaxSelectedObjects(bool includeClusters)
{
    _worker.relaxSelectedObjects(includeClusters);
}

void _SimulationFacadeImpl::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    _worker.uniformVelocitiesForSelectedObjects(includeClusters);
}

void _SimulationFacadeImpl::makeSticky(bool includeClusters)
{
    _worker.makeSticky(includeClusters);
}

void _SimulationFacadeImpl::removeStickiness(bool includeClusters)
{
    _worker.removeStickiness(includeClusters);
}

void _SimulationFacadeImpl::setBarrier(bool value, bool includeClusters)
{
    _worker.setBarrier(value, includeClusters);
}

void _SimulationFacadeImpl::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    _worker.colorSelectedObjects(color, includeClusters);
}

void _SimulationFacadeImpl::reconnectSelectedObjects()
{
    _worker.reconnectSelectedObjects();
}

void _SimulationFacadeImpl::setDetached(bool value)
{
    _worker.setDetached(value);
}

void _SimulationFacadeImpl::changeCell(CellDescription const& changedCell)
{
    _worker.changeCell(changedCell);
}

void _SimulationFacadeImpl::changeParticle(ParticleDescription const& changedParticle)
{
    _worker.changeParticle(changedParticle);
}

void _SimulationFacadeImpl::calcTimesteps(uint64_t timesteps)
{
    _worker.calcTimesteps(timesteps);
    _selectionNeedsUpdate = true;
}

void _SimulationFacadeImpl::runSimulation()
{
    _simRunTimePoint = std::chrono::system_clock::now();
    _worker.runSimulation();
}

void _SimulationFacadeImpl::pauseSimulation()
{
    _worker.pauseSimulation();
    _selectionNeedsUpdate = true;

    _realTime = getRealTime();
    _simRunTimePoint.reset();
}

void _SimulationFacadeImpl::applyCataclysm(int power)
{
    _worker.applyCataclysm(power);
}

bool _SimulationFacadeImpl::isSimulationRunning() const
{
    return _worker.isSimulationRunning();
}

void _SimulationFacadeImpl::closeSimulation()
{
    if (_thread) {
        _worker.beginShutdown();
        _thread->join();
        delete _thread;
        _worker.endShutdown();
        _selectionNeedsUpdate = true;
    }
}

uint64_t _SimulationFacadeImpl::getCurrentTimestep() const
{
    return _worker.getCurrentTimestep();
}

void _SimulationFacadeImpl::setCurrentTimestep(uint64_t value)
{
    _worker.setCurrentTimestep(value);
}

std::chrono::milliseconds _SimulationFacadeImpl::getRealTime() const
{
    if (_simRunTimePoint) {
        return _realTime + std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - *_simRunTimePoint);
    } else {
        return _realTime;
    }
}

void _SimulationFacadeImpl::setRealTime(std::chrono::milliseconds const& value)
{
    _realTime = value;
    if (_simRunTimePoint) {
        _simRunTimePoint = std::chrono::system_clock::now();
    }
}

SimulationParameters _SimulationFacadeImpl::getSimulationParameters() const
{
    return _worker.getSimulationParameters();
}

SimulationParameters const& _SimulationFacadeImpl::getOriginalSimulationParameters() const
{
    return _origSettings.simulationParameters;
}

void _SimulationFacadeImpl::setSimulationParameters(SimulationParameters const& parameters, SimulationParametersUpdateConfig const& updateConfig)
{
    _worker.setSimulationParameters(parameters, updateConfig);
}

void _SimulationFacadeImpl::setOriginalSimulationParameters(SimulationParameters const& parameters)
{
    _origSettings.simulationParameters = parameters;
}

GpuSettings _SimulationFacadeImpl::getGpuSettings() const
{
    return _gpuSettings;
}

GpuSettings _SimulationFacadeImpl::getOriginalGpuSettings() const
{
    return _origSettings.gpuSettings;
}

void _SimulationFacadeImpl::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    _gpuSettings = gpuSettings;
    _worker.setGpuSettings_async(gpuSettings);
}

void _SimulationFacadeImpl::applyForce_async(
    RealVector2D const& start,
    RealVector2D const& end,
    RealVector2D const& force,
    float radius)
{
    _worker.applyForce_async(start, end, force, radius);
}

void _SimulationFacadeImpl::switchSelection(RealVector2D const& pos, float radius)
{
    _worker.switchSelection(pos, radius);
}

void _SimulationFacadeImpl::swapSelection(RealVector2D const& pos, float radius)
{
    _worker.swapSelection(pos, radius);
}

SelectionShallowData _SimulationFacadeImpl::getSelectionShallowData()
{
    return _worker.getSelectionShallowData();
}

void _SimulationFacadeImpl::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData)
{
    _worker.shallowUpdateSelectedObjects(updateData);
}

void _SimulationFacadeImpl::setSelection(RealVector2D const& startPos, RealVector2D const& endPos)
{
    _worker.setSelection(startPos, endPos);
}

void _SimulationFacadeImpl::removeSelection()
{
    _worker.removeSelection();
}

bool _SimulationFacadeImpl::updateSelectionIfNecessary()
{
    auto result = _selectionNeedsUpdate;
    _selectionNeedsUpdate = false;
    if (result) {
        _worker.updateSelection();
    }
    return result;
}

IntVector2D _SimulationFacadeImpl::getWorldSize() const
{
    return _worldSize;
}

StatisticsRawData _SimulationFacadeImpl::getStatisticsRawData() const
{
    return _worker.getStatisticsRawData();
}

StatisticsHistory const& _SimulationFacadeImpl::getStatisticsHistory() const
{
    return _worker.getStatisticsHistory();
}

void _SimulationFacadeImpl::setStatisticsHistory(StatisticsHistoryData const& data)
{
    _worker.setStatisticsHistory(data);
}

std::optional<int> _SimulationFacadeImpl::getTpsRestriction() const
{
    auto result = _worker.getTpsRestriction();
    return 0 != result ? std::optional<int>(result) : std::optional<int>();
}

void _SimulationFacadeImpl::setTpsRestriction(std::optional<int> const& value)
{
    _worker.setTpsRestriction(value ? *value : 0);
}

float _SimulationFacadeImpl::getTps() const
{
    return _worker.getTps();
}

void _SimulationFacadeImpl::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    _worker.testOnly_mutate(cellId, mutationType);
}

void _SimulationFacadeImpl::testOnly_mutationCheck(uint64_t cellId)
{
    _worker.testOnly_mutationCheck(cellId);
}

void _SimulationFacadeImpl::testOnly_createConnection(uint64_t cellId1, uint64_t cellId2)
{
    _worker.testOnly_createConnection(cellId1, cellId2);
}

void _SimulationFacadeImpl::testOnly_cleanupAfterTimestep()
{
    _worker.testOnly_cleanupAfterTimestep();
}

void _SimulationFacadeImpl::testOnly_cleanupAfterDataManipulation()
{
    _worker.testOnly_cleanupAfterDataManipulation();
}

void _SimulationFacadeImpl::testOnly_resizeArrays(ObjectArraySizes const& sizeDelta)
{
    _worker.testOnly_resizeArrays(sizeDelta);
}

bool _SimulationFacadeImpl::testOnly_areArraysValid()
{
    return _worker.testOnly_areArraysValid();
}
