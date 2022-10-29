#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"
#include "ClusterProcessor.cuh"
#include "CellFunctionProcessor.cuh"
#include "NerveProcessor.cuh"
#include "NeuronProcessor.cuh"
#include "ConstructorProcessor.cuh"

__global__ void cudaPrepareNextTimestep(SimulationData data, SimulationResult result)
{
    data.prepareForNextTimestep();
}

__global__ void cudaNextTimestep_substep1(SimulationData data)
{
    CellProcessor::init(data);
    CellProcessor::updateMap(data);
    CellProcessor::radiation(data);  //do not use ParticleProcessor in this kernel
    CellProcessor::clearDensityMap(data);
}

__global__ void cudaNextTimestep_substep2(SimulationData data)
{
    CellProcessor::collisions(data);
    CellProcessor::fillDensityMap(data);

    ParticleProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_substep3(SimulationData data)
{
    CellProcessor::checkForces(data);
    CellProcessor::updateVelocities(data);
    CellProcessor::applyMutation(data);

    ParticleProcessor::movement(data);
    ParticleProcessor::collision(data);
}

__global__ void cudaNextTimestep_verletPositionUpdate(SimulationData data)
{
    CellProcessor::verletPositionUpdate(data);
    CellProcessor::checkConnections(data);
}

__global__ void cudaNextTimestep_connectionForces(SimulationData data, bool considerAngles)
{
    CellProcessor::calcConnectionForces(data, considerAngles);
}

__global__ void cudaNextTimestep_verletVelocityUpdate(SimulationData data)
{
    CellProcessor::verletVelocityUpdate(data);
}

__global__ void cudaNextTimestep_collectCellFunctionOperation(SimulationData data)
{
    CellFunctionProcessor::collectCellFunctionOperations(data);
}

__global__ void cudaNextTimestep_nerveFunction(SimulationData data, SimulationResult result)
{
    NerveProcessor::process(data, result);
}

__global__ void cudaNextTimestep_neuronFunction(SimulationData data, SimulationResult result)
{
    NeuronProcessor::process(data, result);
}

__global__ void cudaNextTimestep_constructorFunction(SimulationData data, SimulationResult result)
{
    ConstructorProcessor::process(data, result);
}

__global__ void cudaNextTimestep_innerFriction(SimulationData data)
{
    CellProcessor::applyInnerFriction(data);
}

__global__ void cudaNextTimestep_friction_decay_finishCellFunctions(SimulationData data)
{
    CellFunctionProcessor::resetFetchedActivities(data);
    CellProcessor::applyFriction(data);
    CellProcessor::decay(data);
}

__global__ void cudaNextTimestep_structuralOperations_step1(SimulationData data)
{
    data.structuralOperations.saveNumEntries();
}

__global__ void cudaNextTimestep_structuralOperations_step2(SimulationData data)
{
    CellConnectionProcessor::processConnectionsOperations(data);
}

__global__ void cudaNextTimestep_substep14(SimulationData data)
{
    ParticleProcessor::transformation(data);

    CellConnectionProcessor::processDelCellOperations(data);
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


//This is the only kernel that uses dynamic parallelism.
//When it is removed, performance drops by about 20% for unknown reasons.
__global__ void nestedDummy() {}
__global__ void dummy()
{
    nestedDummy<<<1, 1>>>();
}
