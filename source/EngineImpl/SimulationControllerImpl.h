#pragma once

#include <thread>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SymbolMap.h"
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

    void initCuda() override;

    void newSimulation(uint64_t timestep, Settings const& settings, SymbolMap const& symbolMap) override;
    void clear() override;

    void registerImageResource(void* image) override;

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

    ClusteredDataDescription getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight) override;
    DataDescription getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight) override;
    ClusteredDataDescription getSelectedClusteredSimulationData(bool includeClusters) override;
    DataDescription getSelectedSimulationData(bool includeClusters) override;
    DataDescription getInspectedSimulationData(std::vector<uint64_t> entityIds) override;

    void addAndSelectSimulationData(DataDescription const& dataToAdd) override;
    void setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate) override;
    void setSimulationData(DataDescription const& dataToUpdate) override;
    void removeSelectedEntities(bool includeClusters) override;
    void relaxSelectedEntities(bool includeClusters) override;
    void colorSelectedEntities(unsigned char color, bool includeClusters) override;
    void reconnectSelectedEntities() override;
    void changeCell(CellDescription const& changedCell) override;
    void changeParticle(ParticleDescription const& changedParticle) override;

    void calcSingleTimestep() override;
    void runSimulation() override;
    void pauseSimulation() override;

    bool isSimulationRunning() const override;

    void closeSimulation() override;

    uint64_t getCurrentTimestep() const override;
    void setCurrentTimestep(uint64_t value) override;

    SimulationParameters const& getSimulationParameters() const override;
    SimulationParameters getOriginalSimulationParameters() const override;
    void setSimulationParameters_async(SimulationParameters const& parameters) override;

    SimulationParametersSpots getSimulationParametersSpots() const override;
    SimulationParametersSpots getOriginalSimulationParametersSpots() const override;
    void setOriginalSimulationParametersSpot(SimulationParametersSpot const& value, int index) override;
    void setSimulationParametersSpots_async(SimulationParametersSpots const& value) override;

    GpuSettings getGpuSettings() const override;
    GpuSettings getOriginalGpuSettings() const override;
    void setGpuSettings_async(GpuSettings const& gpuSettings) override;

    FlowFieldSettings getFlowFieldSettings() const override;
    FlowFieldSettings getOriginalFlowFieldSettings() const override;
    void setOriginalFlowFieldCenter(FlowCenter const& value, int index) override;
    void setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings) override;

    void
    applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius) override;

    void switchSelection(RealVector2D const& pos, float radius) override;
    void swapSelection(RealVector2D const& pos, float radius) override;
    SelectionShallowData getSelectionShallowData() override;
    void shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& updateData) override;
    void setSelection(RealVector2D const& startPos, RealVector2D const& endPos) override;
    void removeSelection() override;
    bool updateSelectionIfNecessary() override;

    GeneralSettings getGeneralSettings() const override;
    IntVector2D getWorldSize() const override;
    Settings getSettings() const override;
    SymbolMap const& getSymbolMap() const override;
    SymbolMap const& getOriginalSymbolMap() const override;
    void setSymbolMap(SymbolMap const& symbolMap) override;
    OverallStatistics getStatistics() const override;

    std::optional<int> getTpsRestriction() const override;
    void setTpsRestriction(std::optional<int> const& value) override;

    float getTps() const override;

private:
    bool _selectionNeedsUpdate = false;

    Settings _origSettings;
    Settings _settings;
    SymbolMap _symbolMap;
    SymbolMap _origSymbolMap;

    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
