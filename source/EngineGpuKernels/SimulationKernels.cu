#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"

__global__ void prepareForNextTimestep(SimulationData data, SimulationResult result)
{
    data.prepareForNextTimestep();
    result.resetStatistics();
    result.setArrayResizeNeeded(data.shouldResize());
}

__global__ void processingStep1(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.init(data);
    cellProcessor.clearTag(data);
    cellProcessor.updateMap(data);
    cellProcessor.radiation(data);  //do not use ParticleProcessor in this kernel
}

__global__ void processingStep2(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.collisions(data);

    ParticleProcessor particleProcessor;
    particleProcessor.updateMap(data);
}

__global__ void processingStep3(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.checkForces(data);
    cellProcessor.updateVelocities(data);
    cellProcessor.clearTag(data);

    ParticleProcessor particleProcessor;
    particleProcessor.movement(data);
    particleProcessor.collision(data);
}

__global__ void processingStep4(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcConnectionForces(data);

    TokenProcessor tokenProcessor;
    tokenProcessor.movement(data);  //changes cell energy without lock
}

__global__ void processingStep5(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.verletUpdatePositions(data);
    cellProcessor.checkConnections(data);
}

__global__ void processingStep6(SimulationData data, SimulationResult result)
{
    CellProcessor cellProcessor;
    cellProcessor.calcConnectionForces(data);

    TokenProcessor tokenProcessor;
    tokenProcessor.executeReadonlyCellFunctions(data, result);
}

__global__ void processingStep7(SimulationData data)
{
    SensorProcessor::processVicinitySearches(data);

    CellProcessor cellProcessor;
    cellProcessor.verletUpdateVelocities(data);
}

__global__ void processingStep8(SimulationData data, SimulationResult result)
{
    TokenProcessor tokenProcessor;
    tokenProcessor.executeModifyingCellFunctions(data, result);
}

__global__ void processingStep9(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcAveragedVelocities(data);
}

__global__ void processingStep10(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.applyAveragedVelocities(data);
    cellProcessor.decay(data);
}

__global__ void processingStep11(SimulationData data)
{
    CellConnectionProcessor::processConnectionsOperations(data);
}

__global__ void processingStep12(SimulationData data)
{
    ParticleProcessor particleProcessor;
    particleProcessor.transformation(data);

    CellConnectionProcessor::processDelCellOperations(data);
}

__global__ void processingStep13(SimulationData data)
{
    TokenProcessor tokenProcessor;
    tokenProcessor.deleteTokenIfCellDeleted(data);
}

//This is the only kernel that uses dynamic parallelism.
//When it is removed, performance drops by about 20% for unknown reasons.
__global__ void nestedDummy() {}
__global__ void dummy()
{
    nestedDummy<<<1, 1>>>();
}
