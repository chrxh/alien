#include "DebugKernels.cuh"

__device__ void DEBUG_checkCells(SimulationData& data, float* sumEnergy, int location)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartitionNew(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        if (auto& cell = cells.at(index)) {

            if (reinterpret_cast<uint64_t>(cell) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                || reinterpret_cast<uint64_t>(cell) + sizeof(Cell)
                    >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                printf("wrong cell pointer at %d\n", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }

            if (cell->creature) {
                if (reinterpret_cast<uint64_t>(cell->creature) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                    || reinterpret_cast<uint64_t>(cell->creature) + sizeof(Creature)
                        >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                    printf("wrong creature pointer at %d\n", location);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }
                if (reinterpret_cast<uint64_t>(cell->creature->genome) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                    || reinterpret_cast<uint64_t>(cell->creature->genome) + sizeof(Genome)
                        >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                    printf("wrong genome pointer at %d\n", location);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }
                if (reinterpret_cast<uint64_t>(cell->creature->genome->genes) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                    || reinterpret_cast<uint64_t>(cell->creature->genome->genes) + sizeof(Gene) * cell->creature->genome->numGenes
                        >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                    printf("wrong genes pointer at %d\n", location);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }
                for (int i = 0; i < cell->creature->genome->numGenes; ++i) {
                    auto const& gene = cell->creature->genome->genes[i];
                    if (reinterpret_cast<uint64_t>(gene.nodes) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                        || reinterpret_cast<uint64_t>(gene.nodes) + sizeof(Node) * gene.numNodes
                            >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                        printf("wrong nodes pointer at %d\n", location);
                        CUDA_THROW_NOT_IMPLEMENTED();
                    }
                }
            }

            if (cell->numConnections > MAX_CELL_BONDS) {
                printf("too much cell connections at %d\n", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectingCell = cell->connections[i].cell;
                if (reinterpret_cast<uint64_t>(connectingCell) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                    || reinterpret_cast<uint64_t>(connectingCell) + sizeof(Cell)
                        >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                    printf("wrong connectingCell pointer (cell: %llu, numConnections: %d) at %d\n", cell->id, cell->numConnections, location);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }

                auto displacement = connectingCell->pos - cell->pos;
                data.cellMap.correctDirection(displacement);
                auto actualDistance = Math::length(displacement);
                if (actualDistance > 14) {
                    printf("distance too large at %d\n", location);
                    CUDA_THROW_NOT_IMPLEMENTED();
                }
            }
            if (cell->usableEnergy < 0 || isnan(cell->usableEnergy)) {
                printf("cell energy invalid at %d", location);
                //CUDA_THROW_NOT_IMPLEMENTED();
            }
            if (sumEnergy != nullptr) {
                atomicAdd(sumEnergy, cell->usableEnergy);
            }
        }
    }
}

__device__ void DEBUG_checkParticles(SimulationData& data, float* sumEnergy, int location)
{
    auto partition = calcSystemThreadPartitionNew(data.objects.particles.getNumEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        if (auto& particle = data.objects.particles.at(particleIndex)) {
            if (reinterpret_cast<uint64_t>(particle) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                || reinterpret_cast<uint64_t>(particle) + sizeof(Particle)
                    >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                printf("wrong particle pointer at %d\n", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }
            if (particle->energy < 0 || isnan(particle->energy)) {
                printf("particle energy invalid at %d", location);
                CUDA_THROW_NOT_IMPLEMENTED();
            }
            if (sumEnergy != nullptr) {
                atomicAdd(sumEnergy, particle->energy);
            }
        }
    }
}

__global__ void DEBUG_checkAngles(SimulationData data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartitionNew(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
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

//__global__ void DEBUG_kernel(SimulationData data, int location)
//{
//    float* sumEnergy = new float;
//    *sumEnergy = 0;
//
//    DEPRECATED_KERNEL_CALL_SYNC(DEBUG_checkCellsAndParticles, data, sumEnergy, location);
//
//    float const expectedEnergy = 187500;
//    if (abs(*sumEnergy - expectedEnergy) > 1) {
//        printf("location: %d, actual energy: %f, expected energy: %f\n", location, *sumEnergy, expectedEnergy);
//        CUDA_THROW_NOT_IMPLEMENTED();
//    }
//    delete sumEnergy;
//}
