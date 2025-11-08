#pragma once

#include "ArraySizesForGpu.h"
#include "DataPointCollection.h"
#include "Definitions.h"
#include "MutationType.h"
#include "PreviewDescription.h"
#include "SelectionShallowData.h"
#include "SettingsForSimulation.h"
#include "ShallowUpdateSelectionData.h"
#include "SimulationFacade.h"
#include "SimulationParametersUpdateConfig.h"
#include "StatisticsHistory.h"

class _SimulationFacade
{
public:
    virtual ~_SimulationFacade() = default;

    // Static getter/setter for dependency injection
    static void set(SimulationFacade const& instance);
    static SimulationFacade get();

    virtual void newSimulation(uint64_t timestep, IntVector2D const& worldSize, SimulationParameters const& simulationParameters) = 0;
    virtual int getSessionId() const = 0;

    virtual void clear() = 0;

    virtual std::string getGpuName() const = 0;

    /**
     * Transfers the simulation data from CUDA to the provided buffer.
     * Resizes buffers if necessary and fills it with data for rendering.
     * If the GPU is busy for a specified duration, the buffers will not be updated and std::nullopt will be returned.
     */
    virtual void tryCopyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers, RealRect const& visibleWorldRect) = 0;

    virtual bool isSyncSimulationWithRendering() const = 0;
    virtual void setSyncSimulationWithRendering(bool value) = 0;
    virtual int getSyncSimulationWithRenderingRatio() const = 0;
    virtual void setSyncSimulationWithRenderingRatio(int value) = 0;

    virtual Description getSimulationData() = 0;
    virtual Description getSelectedSimulationData(bool includeClusters) = 0;
    virtual Description getInspectedSimulationData(std::vector<uint64_t> objectsIds) = 0;

    virtual void addAndSelectSimulationData(Description&& dataToAdd) = 0;
    virtual void setSimulationData(Description const& dataToUpdate) = 0;
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
    virtual bool changeCreature(uint64_t creatureId, GenomeDescription const& genome) = 0;

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
    virtual void setSimulationParameters(
        SimulationParameters const& parameters,
        SimulationParametersUpdateConfig const& updateConfig = SimulationParametersUpdateConfig::All) = 0;
    virtual void setOriginalSimulationParameters(SimulationParameters const& parameters) = 0;

    virtual CudaSettings getGpuSettings() const = 0;
    virtual CudaSettings getOriginalGpuSettings() const = 0;
    virtual void setGpuSettings_async(CudaSettings const& gpuSettings) = 0;

    virtual void applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius) = 0;

    virtual void switchSelection(RealVector2D const& pos, float radius) = 0;
    virtual void swapSelection(RealVector2D const& pos, float radius) = 0;
    virtual SelectionShallowData getSelectionShallowData() = 0;
    virtual void shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData) = 0;
    virtual void setSelection(RealVector2D const& startPos, RealVector2D const& endPos) = 0;
    virtual void removeSelection() = 0;
    virtual bool updateSelectionIfNecessary() = 0;

    virtual IntVector2D getWorldSize() const = 0;
    virtual StatisticsRawData getStatisticsRawData() const = 0;
    virtual StatisticsHistory const& getStatisticsHistory() const = 0;
    virtual void setStatisticsHistory(StatisticsHistoryData const& data) = 0;

    virtual std::optional<int> getTpsRestriction() const = 0;
    virtual void setTpsRestriction(std::optional<int> const& value) = 0;

    virtual float getTps() const = 0;

    // Simulated preview
    virtual Description getPreviewData() = 0;
    virtual void setPreviewData(Description const& description) = 0;
    virtual void calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation = false) = 0;
    virtual void calcTimestepsForPreview(int numSteps, bool detailSimulation = false) = 0;
    virtual uint64_t getCurrentTimestepForPreview() = 0;
    virtual void setCurrentTimestepForPreview(uint64_t timestep) = 0;

    // Only for tests
    virtual void testOnly_mutate(uint64_t cellId, MutationType mutationType) = 0;
    virtual void testOnly_mutationCheck(uint64_t cellId) = 0;
    virtual void testOnly_createConnection(uint64_t cellId1, uint64_t cellId2) = 0;
    virtual void testOnly_cleanupAfterTimestep() = 0;
    virtual void testOnly_cleanupAfterDataManipulation() = 0;
    virtual void testOnly_resizeArrays(ArraySizesForGpu const& sizeDelta) = 0;
    virtual bool testOnly_areArraysValid() = 0;

private:
    static SimulationFacade _instance;
};
