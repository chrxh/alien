#pragma once

#include <thread>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SymbolMap.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineWorker.h"

#include "Definitions.h"

class _SimulationController
{
public:

    ENGINEIMPL_EXPORT void initCuda();

    ENGINEIMPL_EXPORT void newSimulation(uint64_t timestep, Settings const& settings, SymbolMap const& symbolMap);
    ENGINEIMPL_EXPORT void clear();

    ENGINEIMPL_EXPORT void registerImageResource(GLuint image);

    /**
     * Draws section of simulation to registered texture.
     * If the GPU is busy for specific time, the texture will not be updated.
     */
    ENGINEIMPL_EXPORT void tryDrawVectorGraphics(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);
    ENGINEIMPL_EXPORT boost::optional<OverlayDescription> tryDrawVectorGraphicsAndReturnOverlay(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);

    ENGINEIMPL_EXPORT DataDescription
    getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    ENGINEIMPL_EXPORT DataDescription getSelectedSimulationData(bool includeClusters);
    ENGINEIMPL_EXPORT DataDescription getInspectedSimulationData(std::vector<uint64_t> entityIds);

    ENGINEIMPL_EXPORT void addAndSelectSimulationData(DataDescription const& dataToAdd);
    ENGINEIMPL_EXPORT void setSimulationData(DataDescription const& dataToUpdate);
    ENGINEIMPL_EXPORT void removeSelectedEntities(bool includeClusters);
    ENGINEIMPL_EXPORT void colorSelectedEntities(unsigned char color, bool includeClusters);
    ENGINEIMPL_EXPORT void changeCell(CellDescription const& changedCell);
    ENGINEIMPL_EXPORT void changeParticle(ParticleDescription const& changedParticle);

    ENGINEIMPL_EXPORT void calcSingleTimestep();
    ENGINEIMPL_EXPORT void runSimulation();
    ENGINEIMPL_EXPORT void pauseSimulation();

    ENGINEIMPL_EXPORT bool isSimulationRunning() const;

    ENGINEIMPL_EXPORT void closeSimulation();

    ENGINEIMPL_EXPORT uint64_t getCurrentTimestep() const;
    ENGINEIMPL_EXPORT void setCurrentTimestep(uint64_t value);

    ENGINEIMPL_EXPORT SimulationParameters const& getSimulationParameters() const;
    ENGINEIMPL_EXPORT SimulationParameters getOriginalSimulationParameters() const;
    ENGINEIMPL_EXPORT void setSimulationParameters_async(SimulationParameters const& parameters);

    ENGINEIMPL_EXPORT SimulationParametersSpots getSimulationParametersSpots() const;
    ENGINEIMPL_EXPORT SimulationParametersSpots getOriginalSimulationParametersSpots() const;
    ENGINEIMPL_EXPORT void setOriginalSimulationParametersSpot(SimulationParametersSpot const& value, int index);
    ENGINEIMPL_EXPORT void setSimulationParametersSpots_async(SimulationParametersSpots const& value);

    ENGINEIMPL_EXPORT GpuSettings getGpuSettings() const;
    ENGINEIMPL_EXPORT GpuSettings getOriginalGpuSettings() const;
    ENGINEIMPL_EXPORT void setGpuSettings_async(GpuSettings const& gpuSettings);

    ENGINEIMPL_EXPORT FlowFieldSettings getFlowFieldSettings() const;
    ENGINEIMPL_EXPORT FlowFieldSettings getOriginalFlowFieldSettings() const;
    ENGINEIMPL_EXPORT void setOriginalFlowFieldCenter(FlowCenter const& value, int index);
    ENGINEIMPL_EXPORT void setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings);

    ENGINEIMPL_EXPORT void
    applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius);

    ENGINEIMPL_EXPORT void switchSelection(RealVector2D const& pos, float radius);
    ENGINEIMPL_EXPORT void swapSelection(RealVector2D const& pos, float radius);
    ENGINEIMPL_EXPORT SelectionShallowData getSelectionShallowData();
    ENGINEIMPL_EXPORT void shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& updateData);
    ENGINEIMPL_EXPORT void setSelection(RealVector2D const& startPos, RealVector2D const& endPos);
    ENGINEIMPL_EXPORT void removeSelection();
    ENGINEIMPL_EXPORT bool updateSelectionIfNecessary();

    ENGINEIMPL_EXPORT GeneralSettings getGeneralSettings() const;
    ENGINEIMPL_EXPORT IntVector2D getWorldSize() const;
    ENGINEIMPL_EXPORT Settings getSettings() const;
    ENGINEIMPL_EXPORT SymbolMap const& getSymbolMap() const;
    ENGINEIMPL_EXPORT OverallStatistics getStatistics() const;

    ENGINEIMPL_EXPORT boost::optional<int> getTpsRestriction() const;
    ENGINEIMPL_EXPORT void setTpsRestriction(boost::optional<int> const& value);

    ENGINEIMPL_EXPORT float getTps() const;

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
