#include "AttackerProcessor.cuh"
#include "ObjectProcessor.cuh"
#include "ClusterProcessor.cuh"
#include "CommunicatorProcessor.cuh"
#include "ConstructorProcessor.cuh"
#include "DepotProcessor.cuh"
#include "DetonatorProcessor.cuh"
#include "DigestorProcessor.cuh"
#include "ForceFieldKernels.cuh"
#include "GeneratorProcessor.cuh"
#include "InjectorProcessor.cuh"
#include "MemoryProcessor.cuh"
#include "MuscleProcessor.cuh"
#include "NeuronProcessor.cuh"
#include "EnergyProcessor.cuh"
#include "ReconnectorProcessor.cuh"
#include "SensorProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationKernels.cuh"

__global__ void cudaNextTimestep_prepare(SimulationData data)
{
    data.objectMap.reset();
    data.energyMap.reset();
    data.processMemory.reset();

    // Heuristics
    auto maxStructureOperations = 1000 + data.entities.objects.getNumEntries() / 2;
    auto maxCellTypeOperations = data.entities.objects.getNumEntries();

    data.structuralOperations.setMemory(data.processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    for (int i = CellType_Base; i < CellType_Count; ++i) {
        data.cellTypeOperations[i].setMemory(data.processMemory.getTypedSubArray<CellTypeOperation>(maxCellTypeOperations), maxCellTypeOperations);
    }
    *data.externalEnergy = cudaSimulationParameters.externalEnergy.value;

    data.entities.saveNumEntries();
}

__global__ void cudaNextTimestep_physics_init(SimulationData data)
{
    ObjectProcessor::init(data);
    EnergyProcessor::calcActiveSources(data);
}

__global__ void cudaNextTimestep_physics_fillMaps(SimulationData data)
{
    ObjectProcessor::updateMap(data);
    ObjectProcessor::radiation(data);  //do not use EnergyParticleProcessor in this calcKernel
    ObjectProcessor::clearDensityMap(data);
}

__global__ void cudaNextTimestep_physics_calcFluidForces(SimulationData data)
{
    ObjectProcessor::calcFluidForces_reconnectCells_correctOverlap(data);
    ObjectProcessor::fillDensityMap(data);
    EnergyProcessor::fillDensityMap(data);

    EnergyProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_calcCollisionForces(SimulationData data)
{
    ObjectProcessor::calcCollisions_reconnectCells_correctOverlap(data);
    ObjectProcessor::fillDensityMap(data);
    EnergyProcessor::fillDensityMap(data);

    EnergyProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_applyForces(SimulationData data)
{
    ObjectProcessor::checkForces(data);
    ObjectProcessor::applyForces(data);

    EnergyProcessor::movement(data);
    EnergyProcessor::collision(data);
}

__global__ void cudaNextTimestep_physics_verletPositionUpdate(SimulationData data)
{
    ObjectProcessor::verletPositionUpdate(data);
    ObjectProcessor::checkConnections(data);

    EnergyProcessor::splitting(data);
}

__global__ void cudaNextTimestep_physics_calcConnectionForces(SimulationData data, bool considerAngles)
{
    ObjectProcessor::calcConnectionForces(data, considerAngles);
}

__global__ void cudaNextTimestep_physics_verletVelocityUpdate(SimulationData data)
{
    ObjectProcessor::verletVelocityUpdate(data);
}

__global__ void cudaNextTimestep_signal_calcFutureSignals(SimulationData data)
{
    SignalProcessor::calcFutureSignals(data);
}

__global__ void cudaNextTimestep_signal_updateSignals(SimulationData data)
{
    SignalProcessor::updateSignals(data);
}

__global__ void cudaNextTimestep_signal_neuralNetworks(SimulationData data, SimulationStatistics statistics)
{
    NeuronProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_energyFlow(SimulationData data)
{
    ObjectProcessor::performEnergyFlow(data);
}

__global__ void cudaNextTimestep_cellType_prepare_substep1(SimulationData data)
{
    ObjectProcessor::aging(data);
    ObjectProcessor::cellStateTransition_calcFutureState(data);
    ObjectProcessor::frontAngleUpdate_calcFutureValue(data);
}

__global__ void cudaNextTimestep_cellType_prepare_substep2(SimulationData data)
{
    SignalProcessor::collectCellTypeOperations(data);
    ObjectProcessor::cellStateTransition_applyNextState(data);
    ObjectProcessor::frontAngleUpdate_applyFutureValue(data);
    ObjectProcessor::updateCellEvents(data);
}

__global__ void cudaNextTimestep_cellType_generator(SimulationData data, SimulationStatistics statistics)
{
    GeneratorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_constructor_completenessCheck(SimulationData data, SimulationStatistics statistics)
{
    ConstructorProcessor::preprocess(data);
}

__global__ void cudaNextTimestep_cellType_constructor(SimulationData data, SimulationStatistics statistics, bool forPreview)
{
    ConstructorProcessor::process(data, statistics, forPreview);
}

__global__ void cudaNextTimestep_cellType_injector(SimulationData data, SimulationStatistics statistics)
{
    InjectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_attacker(SimulationData data, SimulationStatistics statistics)
{
    AttackerProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_depot(SimulationData data, SimulationStatistics statistics)
{
    DepotProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_muscle(SimulationData data, SimulationStatistics statistics)
{
    MuscleProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_sensor(SimulationData data, SimulationStatistics statistics)
{
    SensorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_reconnector(SimulationData data, SimulationStatistics statistics)
{
    ReconnectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_detonator(SimulationData data, SimulationStatistics statistics)
{
    DetonatorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_digestor(SimulationData data, SimulationStatistics statistics)
{
    DigestorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_memory(SimulationData data, SimulationStatistics statistics)
{
    MemoryProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_communicator(SimulationData data, SimulationStatistics statistics)
{
    CommunicatorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_physics_applyInnerFriction(SimulationData data)
{
    ObjectProcessor::applyInnerFriction(data);
}

__global__ void cudaNextTimestep_physics_applyFriction(SimulationData data)
{
    ObjectProcessor::applyFriction(data);
    ObjectProcessor::decay(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep1(SimulationData data)
{
    data.structuralOperations.saveNumEntries();
}

__global__ void cudaNextTimestep_structuralOperations_substep2(SimulationData data)
{
    ObjectConnectionProcessor::processAddOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep3(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteCellOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep4(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteConnectionOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep5(SimulationData data)
{
    EnergyProcessor::transformation(data);
}

__global__ void cudaNextTimestep_incTimestep(SimulationData data)
{
    ++(*data.timestep);
}

__global__ void cudaInitClusterData(SimulationData data)
{
    ClusterProcessor::initClusterData(data);
}

__global__ void cudaFindClusterIteration(SimulationData data)
{
    ClusterProcessor::findClusterIteration(data);
}

__global__ void cudaFindClusterBoundaries(SimulationData data)
{
    ClusterProcessor::findClusterBoundaries(data);
}

__global__ void cudaAccumulateClusterPosAndVel(SimulationData data)
{
    ClusterProcessor::accumulateClusterPosAndVel(data);
}

__global__ void cudaAccumulateClusterAngularProp(SimulationData data)
{
    ClusterProcessor::accumulateClusterAngularProp(data);
}

__global__ void cudaApplyClusterData(SimulationData data)
{
    ClusterProcessor::applyClusterData(data);
}

__global__ void cudaResetDensity(SimulationData data)
{
    ObjectProcessor::resetDensity(data);
}
