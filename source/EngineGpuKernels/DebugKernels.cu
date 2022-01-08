#include "DebugKernels.cuh"

__device__ void DEBUG_checkCells(SimulationData& data, float* sumEnergy, int location)
{
    auto& cells = data.entities.cellPointers;
    auto partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (auto& cell = cells.at(index)) {

            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectingCell = cell->connections[i].cell;

                auto displacement = connectingCell->absPos - cell->absPos;
                data.cellMap.mapDisplacementCorrection(displacement);
                auto actualDistance = Math::length(displacement);
                if (actualDistance > 14) {
                    printf("distance too large at %d\n", location);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }
            }
            if (cell->energy < 0 || isnan(cell->energy)) {
                printf("cell energy invalid at %d", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }
            atomicAdd(sumEnergy, cell->energy);
        }
    }
}

__device__ void DEBUG_checkParticles(SimulationData& data, float* sumEnergy, int location)
{
    auto partition = calcPartition(data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.entities.particlePointers.at(particleIndex)) {
            if (particle->energy < 0 || isnan(particle->energy)) {
                printf("particle energy invalid at %d", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }
            atomicAdd(sumEnergy, particle->energy);
        }
    }
}

__global__ void DEBUG_checkCellsAndParticles(SimulationData data, float* sumEnergy, int location)
{
    DEBUG_checkCells(data, sumEnergy, location);
    DEBUG_checkParticles(data, sumEnergy, location);
}

__global__ void DEBUG_kernel(SimulationData data, int location)
{
    float* sumEnergy = new float;
    *sumEnergy = 0;

    DEPRECATED_KERNEL_CALL_SYNC(DEBUG_checkCellsAndParticles, data, sumEnergy, location);

    float const expectedEnergy = 187500;
    if (abs(*sumEnergy - expectedEnergy) > 1) {
        printf("location: %d, actual energy: %f, expected energy: %f\n", location, *sumEnergy, expectedEnergy);
        CUDA_THROW_NOT_IMPLEMENTED();
    }
    delete sumEnergy;
}
