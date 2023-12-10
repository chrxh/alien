#pragma once

#include <thread>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/SimulationController.h"
#include "EngineWorker.h"

#include "Definitions.h"

class _SimulationControllerImpl : public _SimulationController
{
public:
    void newSimulation(uint64_t timestep, GeneralSettings const& generalSettings, SimulationParameters const& parameters) override;
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

    ClusteredDataDescription getClusteredSimulationData() override;
    DataDescription getSimulationData() override;
    ClusteredDataDescription getSelectedClusteredSimulationData(bool includeClusters) override;
    DataDescription getSelectedSimulationData(bool includeClusters) override;
    DataDescription getInspectedSimulationData(std::vector<uint64_t> objectIds) override;

    void addAndSelectSimulationData(DataDescription const& dataToAdd) override;
    void setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate) override;
    void setSimulationData(DataDescription const& dataToUpdate) override;
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

    SimulationParameters getSimulationParameters() const override;
    SimulationParameters const& getOriginalSimulationParameters() const override;
    void setSimulationParameters(SimulationParameters const& parameters) override;
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

    GeneralSettings getGeneralSettings() const override;
    IntVector2D getWorldSize() const override;
    RawStatisticsData getRawStatistics() const override;
    StatisticsHistory const& getStatisticsHistory() const override;
    void setStatisticsHistory(StatisticsHistoryData const& data) override;

    std::optional<int> getTpsRestriction() const override;
    void setTpsRestriction(std::optional<int> const& value) override;

    float getTps() const override;

    //for tests
    void testOnly_mutate(uint64_t cellId, MutationType mutationType) override;

private:
    bool _selectionNeedsUpdate = false;

    Settings _origSettings;
    GeneralSettings _generalSettings;
    GpuSettings _gpuSettings;

    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
