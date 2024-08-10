#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"
#include "ClusterProcessor.cuh"
#include "CellFunctionProcessor.cuh"
#include "NerveProcessor.cuh"
#include "NeuronProcessor.cuh"
#include "ConstructorProcessor.cuh"
#include "AttackerProcessor.cuh"
#include "InjectorProcessor.cuh"
#include "TransmitterProcessor.cuh"
#include "MuscleProcessor.cuh"
#include "SensorProcessor.cuh"
#include "CellProcessor.cuh"
#include "ParticleProcessor.cuh"
#include "ReconnectorProcessor.cuh"
#include "DetonatorProcessor.cuh"

__global__ void cudaNextTimestep_prepare(SimulationData data, SimulationStatistics statistics)
{
    data.prepareForNextTimestep();
}

__global__ void cudaNextTimestep_physics_init(SimulationData data)
{
    CellProcessor::init(data);
}

__global__ void cudaNextTimestep_physics_fillMaps(SimulationData data)
{
    CellProcessor::updateMap(data);
    CellProcessor::radiation(data);  //do not use ParticleProcessor in this calcKernel
    CellProcessor::clearDensityMap(data);
}

__global__ void cudaNextTimestep_physics_calcFluidForces(SimulationData data)
{
    CellProcessor::calcFluidForces_reconnectCells_correctOverlap(data);
    CellProcessor::fillDensityMap(data);

    ParticleProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_calcCollisionForces(SimulationData data)
{
    CellProcessor::calcCollisions_reconnectCells_correctOverlap(data);
    CellProcessor::fillDensityMap(data);

    ParticleProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_applyForces(SimulationData data)
{
    CellProcessor::checkForces(data);
    CellProcessor::applyForces(data);

    ParticleProcessor::movement(data);
    ParticleProcessor::collision(data);
}

__global__ void cudaNextTimestep_physics_verletPositionUpdate(SimulationData data)
{
    CellProcessor::verletPositionUpdate(data);
    CellProcessor::checkConnections(data);

    ParticleProcessor::splitting(data);
}

__global__ void cudaNextTimestep_physics_calcConnectionForces(SimulationData data, bool considerAngles)
{
    CellProcessor::calcConnectionForces(data, considerAngles);
}

__global__ void cudaNextTimestep_physics_verletVelocityUpdate(SimulationData data)
{
    CellProcessor::verletVelocityUpdate(data);
}

__global__ void cudaNextTimestep_cellFunction_prepare_substep1(SimulationData data)
{
    CellProcessor::aging(data);
    MutationProcessor::applyRandomMutations(data);
}

__global__ void cudaNextTimestep_cellFunction_prepare_substep2(SimulationData data)
{
    CellProcessor::livingStateTransition(data);
    CellFunctionProcessor::collectCellFunctionOperations(data);
    CellFunctionProcessor::updateRenderingData(data);
}

__global__ void cudaNextTimestep_cellFunction_nerve(SimulationData data, SimulationStatistics statistics)
{
    NerveProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_neuron(SimulationData data, SimulationStatistics statistics)
{
    NeuronProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_constructor_completenessCheck(SimulationData data, SimulationStatistics statistics)
{
    ConstructorProcessor::preprocess(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_constructor_process(SimulationData data, SimulationStatistics statistics)
{
    ConstructorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_injector(SimulationData data, SimulationStatistics statistics)
{
    InjectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_attacker(SimulationData data, SimulationStatistics statistics)
{
    AttackerProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_transmitter(SimulationData data, SimulationStatistics statistics)
{
    TransmitterProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_muscle(SimulationData data, SimulationStatistics statistics)
{
    MuscleProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_sensor(SimulationData data, SimulationStatistics statistics)
{
    SensorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_reconnector(SimulationData data, SimulationStatistics statistics)
{
    ReconnectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellFunction_detonator(SimulationData data, SimulationStatistics statistics)
{
    DetonatorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_physics_applyInnerFriction(SimulationData data)
{
    CellProcessor::applyInnerFriction(data);
}

__global__ void cudaNextTimestep_physics_applyFriction(SimulationData data)
{
    CellFunctionProcessor::resetFetchedActivities(data);
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
    ParticleProcessor::transformation(data);
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
