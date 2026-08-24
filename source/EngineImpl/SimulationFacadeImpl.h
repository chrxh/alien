#pragma once

#include <chrono>
#include <thread>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/SelectionShallowData.h>
#include <EngineInterface/SettingsForSimulation.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "EngineWorker.h"

class _SimulationFacadeImpl : public _SimulationFacade
{
public:
    static void set(SimulationFacade const& instance);

    ~_SimulationFacadeImpl() override;

    void newSimulation(uint64_t timestep, IntVector2D const& worldSize, SimulationParameters const& parameters) override;
    int getSessionId() const override;

    void clear() override;

    std::string getGpuName() const override;

    void tryCopyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers, RealRect const& visibleWorldRect) override;

    bool isSyncSimulationWithRendering() const override;
    void setSyncSimulationWithRendering(bool value) override;
    int getSyncSimulationWithRenderingRatio() const override;
    void setSyncSimulationWithRenderingRatio(int value) override;

    ContentDesc getSimulationData() override;
    ContentDesc getSelectedSimulationData(bool includeClusters) override;
    ContentDesc getInspectedSimulationData(std::vector<uint64_t> objectIds) override;

    void addAndSelectSimulationData(ContentDesc&& dataToAdd) override;
    void setSimulationData(ContentDesc const& dataToUpdate) override;
    void removeSelectedObjects(bool includeClusters) override;
    void relaxSelectedObjects(bool includeClusters) override;
    void uniformVelocitiesForSelectedObjects(bool includeClusters) override;
    void makeSticky(bool includeClusters) override;
    void removeStickiness(bool includeClusters) override;
    void setBarrier(bool value, bool includeClusters) override;
    void colorSelectedObjects(unsigned char color, bool includeClusters) override;
    void reconnectSelectedObjects() override;
    void setDetached(bool value) override;
    void changeCell(ExtendedObjectDesc const& changedCell) override;
    void changeParticle(EnergyDesc const& changedParticle) override;
    int injectGenomeToSelectedCreatures(GenomeDesc const& genome) override;

    void calcTimesteps(uint64_t timesteps) override;
    void runSimulation() override;
    void pauseSimulation() override;
    void applyCataclysm(int power) override;

    bool isSimulationRunning() const override;
    void checkAndThrowException() const override;

    void closeSimulation() override;

    uint64_t getCurrentTimestep() const override;
    void setCurrentTimestep(uint64_t value) override;

    std::chrono::milliseconds getRealTime() const override;
    void setRealTime(std::chrono::milliseconds const& value) override;

    SimulationParameters getSimulationParameters() const override;
    SimulationParameters const& getOriginalSimulationParameters() const override;
    void setSimulationParameters(
        SimulationParameters const& parameters,
        SimulationParametersUpdateConfig const& updateConfig = SimulationParametersUpdateConfig::All) override;
    void setOriginalSimulationParameters(SimulationParameters const& parameters) override;

    void setDebugMode(bool value) override;

    void applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius) override;

    void switchSelection(RealVector2D const& pos, float radius) override;
    void swapSelection(RealVector2D const& pos, float radius) override;
    SelectionShallowData getSelectionShallowData() override;
    void shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData) override;
    void setSelection(RealVector2D const& startPos, RealVector2D const& endPos) override;
    void removeSelection() override;
    bool updateSelectionIfNecessary() override;

    IntVector2D getWorldSize() const override;
    StatisticsHistory const& getStatisticsHistory() const override;
    void setStatisticsHistory(StatisticsHistoryData const& data) override;
    StatisticsEntry getStatisticsEntry() const override;

    std::optional<int> getTpsRestriction() const override;
    void setTpsRestriction(std::optional<int> const& value) override;

    float getTps() const override;

    // Simulated preview
    ContentDesc getPreviewData() override;
    void setPreviewData(ContentDesc const& description) override;
    void calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation) override;
    void calcTimestepsForPreview(int numSteps, bool detailSimulation) override;
    uint64_t getCurrentTimestepForPreview() override;
    void setCurrentTimestepForPreview(uint64_t timestep) override;

    // For tests only
    void testOnly_mutate(uint64_t objectId) override;
    void testOnly_voidUnreachableNodes(uint64_t objectId) override;
    void testOnly_removeUnusedGenes(uint64_t objectId) override;
    void testOnly_removeGeneCycles(uint64_t objectId) override;
    void testOnly_limitGenesWithSeparation(uint64_t objectId) override;
    void testOnly_createConnection(uint64_t objectId1, uint64_t objectId2) override;
    void testOnly_createConnectionWithAbsAngle(uint64_t objectId1, uint64_t objectId2, float desiredDistance, float desiredAbsAngle1, float desiredAbsAngle2)
        override;
    void testOnly_cleanupAfterTimestep() override;
    void testOnly_cleanupAfterDataManipulation() override;
    void testOnly_resizeArrays(ArraySizesForGpuEntities const& sizeDelta) override;
    bool testOnly_isDataValid() override;
    void testOnly_calcTimestepWithCellFunctions() override;
    void testOnly_calcTimestepWithCellFunctionsForPreview(bool detailSimulation = false) override;
    void testOnly_zeroTransferData() override;
    void testOnly_syncNumberGenerator() override;

private:
    bool _selectionNeedsUpdate = false;
    int _sessionId = 0;

    IntVector2D _worldSize;

    SettingsForSimulation _origSettings;

    std::chrono::milliseconds _realTime;
    std::optional<std::chrono::time_point<std::chrono::system_clock>> _simRunTimePoint;

    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
