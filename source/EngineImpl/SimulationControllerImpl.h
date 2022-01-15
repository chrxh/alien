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

    ENGINEIMPL_EXPORT void initCuda() override;

    ENGINEIMPL_EXPORT void newSimulation(uint64_t timestep, Settings const& settings, SymbolMap const& symbolMap) override;
    ENGINEIMPL_EXPORT void clear() override;

    ENGINEIMPL_EXPORT void registerImageResource(void* image) override;

    /**
     * Draws section of simulation to registered texture.
     * If the GPU is busy for specific time, the texture will not be updated.
     */
    ENGINEIMPL_EXPORT void tryDrawVectorGraphics(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom) override;
    ENGINEIMPL_EXPORT std::optional<OverlayDescription>
    tryDrawVectorGraphicsAndReturnOverlay(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom) override;

    ENGINEIMPL_EXPORT ClusteredDataDescription getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight) override;
    ENGINEIMPL_EXPORT DataDescription getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight) override;
    ENGINEIMPL_EXPORT ClusteredDataDescription getSelectedClusteredSimulationData(bool includeClusters) override;
    ENGINEIMPL_EXPORT DataDescription getSelectedSimulationData(bool includeClusters) override;
    ENGINEIMPL_EXPORT DataDescription getInspectedSimulationData(std::vector<uint64_t> entityIds) override;

    ENGINEIMPL_EXPORT void addAndSelectSimulationData(DataDescription const& dataToAdd) override;
    ENGINEIMPL_EXPORT void setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate) override;
    ENGINEIMPL_EXPORT void setSimulationData(DataDescription const& dataToUpdate) override;
    ENGINEIMPL_EXPORT void removeSelectedEntities(bool includeClusters) override;
    ENGINEIMPL_EXPORT void colorSelectedEntities(unsigned char color, bool includeClusters) override;
    ENGINEIMPL_EXPORT void reconnectSelectedEntities() override;
    ENGINEIMPL_EXPORT void changeCell(CellDescription const& changedCell) override;
    ENGINEIMPL_EXPORT void changeParticle(ParticleDescription const& changedParticle) override;

    ENGINEIMPL_EXPORT void calcSingleTimestep() override;
    ENGINEIMPL_EXPORT void runSimulation() override;
    ENGINEIMPL_EXPORT void pauseSimulation() override;

    ENGINEIMPL_EXPORT bool isSimulationRunning() const override;

    ENGINEIMPL_EXPORT void closeSimulation() override;

    ENGINEIMPL_EXPORT uint64_t getCurrentTimestep() const override;
    ENGINEIMPL_EXPORT void setCurrentTimestep(uint64_t value) override;

    ENGINEIMPL_EXPORT SimulationParameters const& getSimulationParameters() const override;
    ENGINEIMPL_EXPORT SimulationParameters getOriginalSimulationParameters() const override;
    ENGINEIMPL_EXPORT void setSimulationParameters_async(SimulationParameters const& parameters) override;

    ENGINEIMPL_EXPORT SimulationParametersSpots getSimulationParametersSpots() const override;
    ENGINEIMPL_EXPORT SimulationParametersSpots getOriginalSimulationParametersSpots() const override;
    ENGINEIMPL_EXPORT void setOriginalSimulationParametersSpot(SimulationParametersSpot const& value, int index) override;
    ENGINEIMPL_EXPORT void setSimulationParametersSpots_async(SimulationParametersSpots const& value) override;

    ENGINEIMPL_EXPORT GpuSettings getGpuSettings() const override;
    ENGINEIMPL_EXPORT GpuSettings getOriginalGpuSettings() const override;
    ENGINEIMPL_EXPORT void setGpuSettings_async(GpuSettings const& gpuSettings) override;

    ENGINEIMPL_EXPORT FlowFieldSettings getFlowFieldSettings() const override;
    ENGINEIMPL_EXPORT FlowFieldSettings getOriginalFlowFieldSettings() const override;
    ENGINEIMPL_EXPORT void setOriginalFlowFieldCenter(FlowCenter const& value, int index) override;
    ENGINEIMPL_EXPORT void setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings) override;

    ENGINEIMPL_EXPORT void
    applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius) override;

    ENGINEIMPL_EXPORT void switchSelection(RealVector2D const& pos, float radius) override;
    ENGINEIMPL_EXPORT void swapSelection(RealVector2D const& pos, float radius) override;
    ENGINEIMPL_EXPORT SelectionShallowData getSelectionShallowData() override;
    ENGINEIMPL_EXPORT void shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& updateData) override;
    ENGINEIMPL_EXPORT void setSelection(RealVector2D const& startPos, RealVector2D const& endPos) override;
    ENGINEIMPL_EXPORT void removeSelection() override;
    ENGINEIMPL_EXPORT bool updateSelectionIfNecessary() override;

    ENGINEIMPL_EXPORT GeneralSettings getGeneralSettings() const override;
    ENGINEIMPL_EXPORT IntVector2D getWorldSize() const override;
    ENGINEIMPL_EXPORT Settings getSettings() const override;
    ENGINEIMPL_EXPORT SymbolMap const& getSymbolMap() const override;
    ENGINEIMPL_EXPORT OverallStatistics getStatistics() const override;

    ENGINEIMPL_EXPORT std::optional<int> getTpsRestriction() const override;
    ENGINEIMPL_EXPORT void setTpsRestriction(std::optional<int> const& value) override;

    ENGINEIMPL_EXPORT float getTps() const override;

private:
    bool _selectionNeedsUpdate = false;

    Settings _origSettings;
    Settings _settings;
    GpuSettings _gpuSettings; 
    GpuSettings _origGpuSettings;
    SymbolMap _symbolMap;

    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
