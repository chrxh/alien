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
    cellProcessor.clearDensityMap(data);
}

__global__ void processingStep2(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.collisions(data);
    cellProcessor.fillDensityMap(data);

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
    SensorProcessor::processScheduledOperation(data);

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
    cellProcessor.calcFriction(data);
}

__global__ void processingStep10(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.applyFriction(data);
    cellProcessor.decay(data);
}

__global__ void processingStep11(SimulationData data)
{
    data.structuralOperations.saveNumEntries();
}

__global__ void processingStep12(SimulationData data)
{
    CellConnectionProcessor::processConnectionsOperations(data);
}

__global__ void processingStep13(SimulationData data)
{
    ParticleProcessor particleProcessor;
    particleProcessor.transformation(data);

    CellConnectionProcessor::processDelCellOperations(data);
}

__global__ void processingStep14(SimulationData data)
{
    TokenProcessor tokenProcessor;
    tokenProcessor.deleteTokenIfCellDeleted(data);
}

__global__ void cudaInitClusterData(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        cell->clusterIndex = index;
        cell->clusterBoundaries = 0;
        cell->clusterPos = {0, 0};
        cell->clusterVel = {0, 0};
        cell->clusterAngularMass = 0;
        cell->clusterAngularMomentum = 0;
        cell->numCellsInCluster = 0;
    }
}

__global__ void cudaFindClusterIteration(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto currentCell = cells.at(index);

        //heuristics to cover connected cells
        for (int i = 0; i < 30; ++i) {
            bool found = false;
            for (int j = 0; j < currentCell->numConnections; ++j) {
                auto candidateCell = currentCell->connections[j].cell;
                auto cellTag = currentCell->clusterIndex;
                auto origTag = atomicMin(&candidateCell->clusterIndex, cellTag);
                if (cellTag < origTag) {
                    currentCell = candidateCell;
                    found = true;
                    break;
                }
            }
            if (!found) {
                break;
            }
        }
    }
}

__global__ void cudaFindClusterBoundaries(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto cell = cells.at(index);
        auto cluster = cells.at(cell->clusterIndex);
        if (cell->absPos.x < data.worldSize.x / 3) {
            atomicOr(&cluster->clusterBoundaries, 1);
        }
        if (cell->absPos.y < data.worldSize.y / 3) {
            atomicOr(&cluster->clusterBoundaries, 2);
        }
    }
}

__global__ void cudaAccumulateClusterPosAndVel(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto cell = cells.at(index);
        auto cluster = cells.at(cell->clusterIndex);
        atomicAdd(&cluster->clusterVel.x, cell->vel.x);
        atomicAdd(&cluster->clusterVel.y, cell->vel.y);

        //topology correction
        auto cellPos = cell->absPos;
        if ((cluster->clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }

        atomicAdd(&cluster->clusterPos.x, cellPos.x);
        atomicAdd(&cluster->clusterPos.y, cellPos.y);

        atomicAdd(&cluster->numCellsInCluster, 1);
    }
}

__global__ void cudaAccumulateClusterAngularProp(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto cell = cells.at(index);
        auto cluster = cells.at(cell->clusterIndex);
        auto clusterVel = cluster->clusterVel / cluster->numCellsInCluster;
        auto clusterPos = cluster->clusterPos / cluster->numCellsInCluster;

        //topology correction
        auto cellPos = cell->absPos;
        if ((cluster->clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularMass = Math::lengthSquared(r);
        auto angularMomentum = Physics::angularMomentum(r, cell->vel - clusterVel);
        atomicAdd(&cluster->clusterAngularMass, angularMass);
        atomicAdd(&cluster->clusterAngularMomentum, angularMomentum);
    }
}

__global__ void cudaApplyClusterData(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto cell = cells.at(index);
        auto cluster = cells.at(cell->clusterIndex);
        auto clusterPos = cluster->clusterPos / cluster->numCellsInCluster;
        auto clusterVel = cluster->clusterVel / cluster->numCellsInCluster;

        auto cellPos = cell->absPos;
        if ((cluster->clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularVel = Physics::angularVelocity(cluster->clusterAngularMomentum, cluster->clusterAngularMass);
        cell->vel = cell->vel * 0.0f + Physics::tangentialVelocity(r, clusterVel, angularVel) * 1.0f;
    }
}


//This is the only kernel that uses dynamic parallelism.
//When it is removed, performance drops by about 20% for unknown reasons.
__global__ void nestedDummy() {}
__global__ void dummy()
{
    nestedDummy<<<1, 1>>>();
}
