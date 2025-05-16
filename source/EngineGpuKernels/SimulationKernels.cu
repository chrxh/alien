#include "SimulationKernels.cuh"
#include "ForceFieldKernels.cuh"
#include "ClusterProcessor.cuh"
#include "SignalProcessor.cuh"
#include "OscillatorProcessor.cuh"
#include "NeuronProcessor.cuh"
#include "ConstructorProcessor.cuh"
#include "AttackerProcessor.cuh"
#include "InjectorProcessor.cuh"
#include "TransmitterProcessor.cuh"
#include "MuscleProcessor.cuh"
#include "SensorProcessor.cuh"
#include "CellProcessor.cuh"
#include "RadiationProcessor.cuh"
#include "ReconnectorProcessor.cuh"
#include "DetonatorProcessor.cuh"

__global__ void cudaNextTimestep_prepare(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();
    data.processMemory.reset();

    // Heuristics
    auto maxStructureOperations = 1000 + data.objects.cells.getNumEntries() / 2;
    auto maxCellTypeOperations = data.objects.cells.getNumEntries();

    data.structuralOperations.setMemory(data.processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    for (int i = CellType_Base; i < CellType_Count; ++i) {
        data.cellTypeOperations[i].setMemory(data.processMemory.getTypedSubArray<CellTypeOperation>(maxCellTypeOperations), maxCellTypeOperations);
    }
    *data.externalEnergy = cudaSimulationParameters.externalEnergy.value;

    data.objects.saveNumEntries();
}

__global__ void cudaNextTimestep_physics_init(SimulationData data)
{
    CellProcessor::init(data);
    RadiationProcessor::calcActiveSources(data);
}

__global__ void cudaNextTimestep_physics_fillMaps(SimulationData data)
{
    CellProcessor::updateMap(data);
    CellProcessor::radiation(data);  //do not use RadiationProcessor in this calcKernel
    CellProcessor::clearDensityMap(data);
}

__global__ void cudaNextTimestep_physics_calcFluidForces(SimulationData data)
{
    CellProcessor::calcFluidForces_reconnectCells_correctOverlap(data);
    CellProcessor::fillDensityMap(data);

    RadiationProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_calcCollisionForces(SimulationData data)
{
    CellProcessor::calcCollisions_reconnectCells_correctOverlap(data);
    CellProcessor::fillDensityMap(data);

    RadiationProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_applyForces(SimulationData data)
{
    CellProcessor::checkForces(data);
    CellProcessor::applyForces(data);

    RadiationProcessor::movement(data);
    RadiationProcessor::collision(data);
}

__global__ void cudaNextTimestep_physics_verletPositionUpdate(SimulationData data)
{
    CellProcessor::verletPositionUpdate(data);
    CellProcessor::checkConnections(data);

    RadiationProcessor::splitting(data);
}

__global__ void cudaNextTimestep_physics_calcConnectionForces(SimulationData data, bool considerAngles)
{
    CellProcessor::calcConnectionForces(data, considerAngles);
}

__global__ void cudaNextTimestep_physics_verletVelocityUpdate(SimulationData data)
{
    CellProcessor::verletVelocityUpdate(data);
}

__global__ void cudaNextTimestep_signal_calcFutureSignals(SimulationData data)
{
    SignalProcessor::calcFutureSignals(data);
}

__global__ void cudaNextTimestep_signal_updateSignals(SimulationData data)
{
    SignalProcessor::updateSignals(data);
    SignalProcessor::collectCellTypeOperations(data);
}

__global__ void cudaNextTimestep_signal_neuralNetworks(SimulationData data, SimulationStatistics statistics)
{
    NeuronProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_energyFlow(SimulationData data)
{
    CellProcessor::applyEnergyFlow(data);
}

__global__ void cudaNextTimestep_cellType_prepare_substep1(SimulationData data)
{
    CellProcessor::aging(data);
    MutationProcessor::applyRandomMutations(data);
    CellProcessor::livingStateTransition_calcFutureState(data);
}

__global__ void cudaNextTimestep_cellType_prepare_substep2(SimulationData data)
{
    CellProcessor::livingStateTransition_applyNextState(data);
    CellProcessor::updateRenderingData(data);
}

__global__ void cudaNextTimestep_cellType_oscillator(SimulationData data, SimulationStatistics statistics)
{
    OscillatorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_constructor_completenessCheck(SimulationData data, SimulationStatistics statistics)
{
    ConstructorProcessor::preprocess(data, statistics);
}

__global__ void cudaNextTimestep_cellType_constructor(SimulationData data, SimulationStatistics statistics)
{
    ConstructorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_injector(SimulationData data, SimulationStatistics statistics)
{
    InjectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_attacker(SimulationData data, SimulationStatistics statistics)
{
    AttackerProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_transmitter(SimulationData data, SimulationStatistics statistics)
{
    TransmitterProcessor::process(data, statistics);
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

__global__ void cudaNextTimestep_physics_applyInnerFriction(SimulationData data)
{
    CellProcessor::applyInnerFriction(data);
}

__global__ void cudaNextTimestep_physics_applyFriction(SimulationData data)
{
    CellProcessor::applyFriction(data);
    CellProcessor::decay(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep1(SimulationData data)
{
    data.structuralOperations.saveNumEntries();
}

__global__ void cudaNextTimestep_structuralOperations_substep2(SimulationData data)
{
    CellConnectionProcessor::processAddOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep3(SimulationData data)
{
    CellConnectionProcessor::processDeleteCellOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep4(SimulationData data)
{
    CellConnectionProcessor::processDeleteConnectionOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep5(SimulationData data)
{
    RadiationProcessor::transformation(data);
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
    CellProcessor::resetDensity(data);
}


//This is the only calcKernel that uses dynamic parallelism.
//When it is removed, performance drops by about 20% for unknown reasons.
__global__ void nestedDummy() {}
__global__ void dummy()
{
    nestedDummy<<<1, 1>>>();
}
