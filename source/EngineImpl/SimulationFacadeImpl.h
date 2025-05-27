#pragma once

#include <thread>
#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SettingsForSimulation.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineWorker.h"

#include "Definitions.h"

class _SimulationFacadeImpl : public _SimulationFacade
{
public:
    void newSimulation(uint64_t timestep, IntVector2D const& worldSize, SimulationParameters const& parameters) override;
    int getSessionId() const override;

    void clear() override;

    void setImageResource(void* image) override;
    std::string getGpuName() const override;

    /**
     * Draws section of simulation to registered texture.
     * If the GPU is busy for specific time, the texture will not be updated.
     */
    void tryDrawVectorGraphics(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom) override;
    std::optional<OverlayDescription>
    tryDrawVectorGraphicsAndReturnOverlay(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom) override;

    bool isSyncSimulationWithRendering() const override;
    void setSyncSimulationWithRendering(bool value) override;
    int getSyncSimulationWithRenderingRatio() const override;
    void setSyncSimulationWithRenderingRatio(int value) override;

    CollectionDescription getSimulationData() override;
    CollectionDescription getSelectedSimulationData(bool includeClusters) override;
    CollectionDescription getInspectedSimulationData(std::vector<uint64_t> objectIds) override;

    void addAndSelectSimulationData(CollectionDescription&& dataToAdd) override;
    void setSimulationData(CollectionDescription const& dataToUpdate) override;
    void removeSelectedObjects(bool includeClusters) override;
    void relaxSelectedObjects(bool includeClusters) override;
    void uniformVelocitiesForSelectedObjects(bool includeClusters) override;
    void makeSticky(bool includeClusters) override;
    void removeStickiness(bool includeClusters) override;
    void setBarrier(bool value, bool includeClusters) override;
    void colorSelectedObjects(unsigned char color, bool includeClusters) override;
    void reconnectSelectedObjects() override;
    void setDetached(bool value) override;
    void changeCell(CellDescription const& changedCell) override;
    void changeParticle(ParticleDescription const& changedParticle) override;

    void calcTimesteps(uint64_t timesteps) override;
    void runSimulation() override;
    void pauseSimulation() override;
    void applyCataclysm(int power) override;

    bool isSimulationRunning() const override;

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

    GpuSettings getGpuSettings() const override;
    GpuSettings getOriginalGpuSettings() const override;
    void setGpuSettings_async(GpuSettings const& gpuSettings) override;

    void applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius) override;

    void switchSelection(RealVector2D const& pos, float radius) override;
    void swapSelection(RealVector2D const& pos, float radius) override;
    SelectionShallowData getSelectionShallowData() override;
    void shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData) override;
    void setSelection(RealVector2D const& startPos, RealVector2D const& endPos) override;
    void removeSelection() override;
    bool updateSelectionIfNecessary() override;

    IntVector2D getWorldSize() const override;
    StatisticsRawData getStatisticsRawData() const override;
    StatisticsHistory const& getStatisticsHistory() const override;
    void setStatisticsHistory(StatisticsHistoryData const& data) override;

    std::optional<int> getTpsRestriction() const override;
    void setTpsRestriction(std::optional<int> const& value) override;

    float getTps() const override;

    // for tests only
    void testOnly_mutate(uint64_t cellId, MutationType mutationType) override;
    void testOnly_mutationCheck(uint64_t cellId) override;
    void testOnly_createConnection(uint64_t cellId1, uint64_t cellId2) override;
    void testOnly_cleanupAfterTimestep() override;
    void testOnly_cleanupAfterDataManipulation() override;
    void testOnly_resizeArrays(ArraySizesForGpu const& sizeDelta) override;
    bool testOnly_areArraysValid() override;

private:
    bool _selectionNeedsUpdate = false;
    int _sessionId = 0;

    IntVector2D _worldSize;
    GpuSettings _gpuSettings;

    SettingsForSimulation _origSettings;

    std::chrono::milliseconds _realTime;
    std::optional<std::chrono::time_point<std::chrono::system_clock>> _simRunTimePoint;

    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
