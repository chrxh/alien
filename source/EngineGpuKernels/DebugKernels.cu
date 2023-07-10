#include "DebugKernels.cuh"

#include "GenomeDecoder.cuh"

__device__ void DEBUG_checkCells(SimulationData& data, float* sumEnergy, int location)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (auto& cell = cells.at(index)) {

            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectingCell = cell->connections[i].cell;

                auto displacement = connectingCell->absPos - cell->absPos;
                data.cellMap.correctDirection(displacement);
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
    auto partition = calcPartition(data.objects.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.objects.particlePointers.at(particleIndex)) {
            if (particle->energy < 0 || isnan(particle->energy)) {
                printf("particle energy invalid at %d", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }
            atomicAdd(sumEnergy, particle->energy);
        }
    }
}

__global__ void DEBUG_checkAngles(SimulationData data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (auto& cell = cells.at(index)) {
            if (cell->numConnections > 0) {
                float sumAngles = 0;
                for (int i = 0; i < cell->numConnections; ++i) {
                    sumAngles += cell->connections[i].angleFromPrevious;
                    if (cell->connections[i].angleFromPrevious < -NEAR_ZERO) {
                        printf("invalid angle: %f\n", cell->connections[i].angleFromPrevious);
                        CUDA_THROW_NOT_IMPLEMENTED();
                    }
                    if (cell->connections[i].angleFromPrevious < NEAR_ZERO) {
                        printf("zero angle\n");
                    }
                }
                if (abs(360.0f - sumAngles) > 0.1f) {
                    printf("invalid angle sum: %f\n", sumAngles);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }
            }
        }
    }
}

__global__ void DEBUG_checkCellsAndParticles(SimulationData data, float* sumEnergy, int location)
{
    DEBUG_checkCells(data, sumEnergy, location);
    DEBUG_checkParticles(data, sumEnergy, location);
}

__global__ void DEBUG_checkGenomes(SimulationData data, int location)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (auto& cell = cells.at(index)) {
            if (cell->cellFunction == CellFunction_Constructor || cell->cellFunction == CellFunction_Injector) {
                auto genome = cell->getGenome();
                auto genomeSize = cell->getGenomeSize();


                if (genomeSize < Const::GenomeHeaderSize) {
                    printf("1: %d\n", location);
                    ABORT();
                }
                int subGenomeEndAddresses[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
                int depth = 0;
                for (auto nodeAddress = Const::GenomeHeaderSize; nodeAddress < genomeSize;) {
                    auto cellFunction = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);

                    bool goToNextSibling = true;
                    if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
                        auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
                        auto makeSelfCopy = GenomeDecoder::convertByteToBool(genome[nodeAddress + Const::CellBasicBytes + cellFunctionFixedBytes]);
                        if (!makeSelfCopy) {
                            auto subGenomeSize = GenomeDecoder::getNextSubGenomeSize(genome, genomeSize, nodeAddress);
                            nodeAddress += Const::CellBasicBytes + cellFunctionFixedBytes + 3;
                            subGenomeEndAddresses[depth++] = nodeAddress + subGenomeSize;
                            nodeAddress += Const::GenomeHeaderSize;
                            goToNextSibling = false;
                            if (nodeAddress > genomeSize) {
                                printf("2: %d\n", location);
                                ABORT();
                            }
                        }
                    }

                    if (goToNextSibling) {
                        nodeAddress += Const::CellBasicBytes + GenomeDecoder::getNextCellFunctionDataSize(genome, genomeSize, nodeAddress);
                        if (nodeAddress > genomeSize) {
                            printf("3: %d\n", location);
                            ABORT();
                        }
                    }
                    for (int i = 0; i < GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH && depth > 0; ++i) {
                        if (depth > 0) {
                            if (subGenomeEndAddresses[depth - 1] == nodeAddress) {
                                --depth;
                            } else {
                                break;
                            }
                        }
                    }
                }

            }
        }
    }
}

/*
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
*/
