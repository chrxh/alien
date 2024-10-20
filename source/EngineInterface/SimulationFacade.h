#pragma once
#include "Definitions.h"
#include "OverlayDescriptions.h"
#include "SelectionShallowData.h"
#include "Settings.h"
#include "ShallowUpdateSelectionData.h"
#include "SimulationFacade.h"
#include "MutationType.h"
#include "DataPointCollection.h"
#include "StatisticsHistory.h"

class _SimulationFacade
{
public:
    virtual ~_SimulationFacade() = default;

    virtual void newSimulation(uint64_t timestep, GeneralSettings const& generalSettings, SimulationParameters const& simulationParameters) = 0;
    virtual int getSessionId() const = 0;

    virtual void clear() = 0;

    virtual void setImageResource(void* image) = 0;
    virtual std::string getGpuName() const = 0;

    /**
     * Draws a section of simulation to a registered texture.
     * If the GPU is busy for a specified duration, the texture will not be updated.
     */
    virtual void tryDrawVectorGraphics(RealVector2D const& rectUpperLeft, RealVector2D const& rectLowerRight, IntVector2D const& imageSize, double zoom) = 0;
    virtual std::optional<OverlayDescription>
    tryDrawVectorGraphicsAndReturnOverlay(RealVector2D const& rectUpperLeft, RealVector2D const& rectLowerRight, IntVector2D const& imageSize, double zoom) = 0;

    virtual bool isSyncSimulationWithRendering() const = 0;
    virtual void setSyncSimulationWithRendering(bool value) = 0;
    virtual int getSyncSimulationWithRenderingRatio() const = 0;
    virtual void setSyncSimulationWithRenderingRatio(int value) = 0;

    virtual ClusteredDataDescription getClusteredSimulationData() = 0;
    virtual DataDescription getSimulationData() = 0;
    virtual ClusteredDataDescription getSelectedClusteredSimulationData(bool includeClusters) = 0;
    virtual DataDescription getSelectedSimulationData(bool includeClusters) = 0;
    virtual DataDescription getInspectedSimulationData(std::vector<uint64_t> objectsIds) = 0;

    virtual void addAndSelectSimulationData(DataDescription const& dataToAdd) = 0;
    virtual void setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate) = 0;
    virtual void setSimulationData(DataDescription const& dataToUpdate) = 0;
    virtual void removeSelectedObjects(bool includeClusters) = 0;
    virtual void relaxSelectedObjects(bool includeClusters) = 0;
    virtual void uniformVelocitiesForSelectedObjects(bool includeClusters) = 0;
    virtual void makeSticky(bool includeClusters) = 0;
    virtual void removeStickiness(bool includeClusters) = 0;
    virtual void setBarrier(bool value, bool includeClusters) = 0;
    virtual void colorSelectedObjects(unsigned char color, bool includeClusters) = 0;
    virtual void reconnectSelectedObjects() = 0;
    virtual void setDetached(bool value) = 0;
    virtual void changeCell(CellDescription const& changedCell) = 0;
    virtual void changeParticle(ParticleDescription const& changedParticle) = 0;

    virtual void calcTimesteps(uint64_t timesteps) = 0;
    virtual void runSimulation() = 0;
    virtual void pauseSimulation() = 0;
    virtual void applyCataclysm(int power) = 0;

    virtual bool isSimulationRunning() const = 0;

    virtual void closeSimulation() = 0;

    virtual uint64_t getCurrentTimestep() const = 0;
    virtual void setCurrentTimestep(uint64_t value) = 0;

    virtual std::chrono::milliseconds getRealTime() const = 0;
    virtual void setRealTime(std::chrono::milliseconds const& value) = 0;

    virtual SimulationParameters getSimulationParameters() const = 0;
    virtual SimulationParameters const& getOriginalSimulationParameters() const = 0;
    virtual void setSimulationParameters(SimulationParameters const& parameters) = 0;
    virtual void setOriginalSimulationParameters(SimulationParameters const& parameters) = 0;

    virtual GpuSettings getGpuSettings() const = 0;
    virtual GpuSettings getOriginalGpuSettings() const = 0;
    virtual void setGpuSettings_async(GpuSettings const& gpuSettings) = 0;

    virtual void applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius) = 0;

    virtual void switchSelection(RealVector2D const& pos, float radius) = 0;
    virtual void swapSelection(RealVector2D const& pos, float radius) = 0;
    virtual SelectionShallowData getSelectionShallowData(RealVector2D const& refPos = RealVector2D()) = 0;
    virtual void shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData) = 0;
    virtual void setSelection(RealVector2D const& startPos, RealVector2D const& endPos) = 0;
    virtual void removeSelection() = 0;
    virtual bool updateSelectionIfNecessary() = 0;

    virtual GeneralSettings getGeneralSettings() const = 0;
    virtual IntVector2D getWorldSize() const = 0;
    virtual RawStatisticsData getRawStatistics() const = 0;
    virtual StatisticsHistory const& getStatisticsHistory() const = 0;
    virtual void setStatisticsHistory(StatisticsHistoryData const& data) = 0;

    virtual std::optional<int> getTpsRestriction() const = 0;
    virtual void setTpsRestriction(std::optional<int> const& value) = 0;

    virtual float getTps() const = 0;

    //for tests
    virtual void testOnly_mutate(uint64_t cellId, MutationType mutationType) = 0;
};
