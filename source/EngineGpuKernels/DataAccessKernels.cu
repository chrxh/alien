#include "DataAccessKernels.cuh"

namespace
{
    template <typename T>
    __device__ void
    copyAuxiliaryData(T sourceSize, uint8_t* source, T& targetSize, uint64_t& targetIndex, uint64_t& auxiliaryDataSize, uint8_t*& auxiliaryData)
    {
        targetSize = sourceSize;
        if (sourceSize > 0) {
            targetIndex = alienAtomicAdd64(&auxiliaryDataSize, static_cast<uint64_t>(sourceSize));
            for (int i = 0; i < sourceSize; ++i) {
                auxiliaryData[targetIndex + i] = source[i];
            }
        }
    }

    __device__ void createCellTO(Cell* cell, DataTO& dataTO, Cell* cellArrayStart)
    {
        auto cellTOIndex = alienAtomicAdd64(dataTO.numCells, uint64_t(1));
        auto& cellTO = dataTO.cells[cellTOIndex];

        cellTO.id = cell->id;
        cellTO.pos = cell->pos;
        cellTO.vel = cell->vel;
        cellTO.barrier = cell->barrier;
        cellTO.energy = cell->energy;
        cellTO.stiffness = cell->stiffness;
        cellTO.maxConnections = cell->maxConnections;
        cellTO.numConnections = cell->numConnections;
        cellTO.executionOrderNumber = cell->executionOrderNumber;
        cellTO.livingState = cell->livingState;
        cellTO.creatureId = cell->creatureId;
        cellTO.mutationId = cell->mutationId;
        cellTO.ancestorMutationId = cell->ancestorMutationId;
        cellTO.genomeComplexity = cell->genomeComplexity;
        cellTO.inputExecutionOrderNumber = cell->inputExecutionOrderNumber;
        cellTO.outputBlocked = cell->outputBlocked;
        cellTO.cellFunction = cell->cellFunction;
        cellTO.color = cell->color;
        cellTO.age = cell->age;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cell->signal.channels[i];
        }
        cellTO.signal.origin = cell->signal.origin;
        cellTO.signal.targetX = cell->signal.targetX;
        cellTO.signal.targetY = cell->signal.targetY;
        cellTO.activationTime = cell->activationTime;
        cellTO.detectedByCreatureId = cell->detectedByCreatureId;
        cellTO.cellFunctionUsed = cell->cellFunctionUsed;

        copyAuxiliaryData(
            cell->metadata.nameSize,
            cell->metadata.name,
            cellTO.metadata.nameSize,
            cellTO.metadata.nameDataIndex,
            *dataTO.numAuxiliaryData,
            dataTO.auxiliaryData);
        copyAuxiliaryData(
            cell->metadata.descriptionSize,
            cell->metadata.description,
            cellTO.metadata.descriptionSize,
            cellTO.metadata.descriptionDataIndex,
            *dataTO.numAuxiliaryData,
            dataTO.auxiliaryData);

        cell->tag = cellTOIndex;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;
            cellTO.connections[i].cellIndex = connectingCell - cellArrayStart;
            cellTO.connections[i].distance = cell->connections[i].distance;
            cellTO.connections[i].angleFromPrevious = cell->connections[i].angleFromPrevious;
        }

        switch (cell->cellFunction) {
        case CellFunction_Neuron: {
            int targetSize;    //not used
            copyAuxiliaryData<int>(
                sizeof(NeuronFunction::NeuronState),
                reinterpret_cast<uint8_t*>(cell->cellFunctionData.neuron.neuronState),
                targetSize,
                cellTO.cellFunctionData.neuron.weightsAndBiasesDataIndex,
                *dataTO.numAuxiliaryData,
                dataTO.auxiliaryData);
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cellTO.cellFunctionData.neuron.activationFunctions[i] = cell->cellFunctionData.neuron.activationFunctions[i];
            }
        } break;
        case CellFunction_Transmitter: {
            cellTO.cellFunctionData.transmitter.mode = cell->cellFunctionData.transmitter.mode;
        } break;
        case CellFunction_Constructor: {
            cellTO.cellFunctionData.constructor.activationMode = cell->cellFunctionData.constructor.activationMode;
            cellTO.cellFunctionData.constructor.constructionActivationTime = cell->cellFunctionData.constructor.constructionActivationTime;
            copyAuxiliaryData(
                cell->cellFunctionData.constructor.genomeSize,
                cell->cellFunctionData.constructor.genome,
                cellTO.cellFunctionData.constructor.genomeSize,
                cellTO.cellFunctionData.constructor.genomeDataIndex,
                *dataTO.numAuxiliaryData,
                dataTO.auxiliaryData);
            cellTO.cellFunctionData.constructor.numInheritedGenomeNodes = cell->cellFunctionData.constructor.numInheritedGenomeNodes;
            cellTO.cellFunctionData.constructor.lastConstructedCellId = cell->cellFunctionData.constructor.lastConstructedCellId;
            cellTO.cellFunctionData.constructor.genomeCurrentNodeIndex = cell->cellFunctionData.constructor.genomeCurrentNodeIndex;
            cellTO.cellFunctionData.constructor.genomeCurrentRepetition = cell->cellFunctionData.constructor.genomeCurrentRepetition;
            cellTO.cellFunctionData.constructor.currentBranch = cell->cellFunctionData.constructor.currentBranch;
            cellTO.cellFunctionData.constructor.offspringCreatureId = cell->cellFunctionData.constructor.offspringCreatureId;
            cellTO.cellFunctionData.constructor.offspringMutationId = cell->cellFunctionData.constructor.offspringMutationId;
            cellTO.cellFunctionData.constructor.genomeGeneration = cell->cellFunctionData.constructor.genomeGeneration;
            cellTO.cellFunctionData.constructor.constructionAngle1 = cell->cellFunctionData.constructor.constructionAngle1;
            cellTO.cellFunctionData.constructor.constructionAngle2 = cell->cellFunctionData.constructor.constructionAngle2;
        } break;
        case CellFunction_Sensor: {
            cellTO.cellFunctionData.sensor.mode = cell->cellFunctionData.sensor.mode;
            cellTO.cellFunctionData.sensor.angle = cell->cellFunctionData.sensor.angle;
            cellTO.cellFunctionData.sensor.minDensity = cell->cellFunctionData.sensor.minDensity;
            cellTO.cellFunctionData.sensor.minRange = cell->cellFunctionData.sensor.minRange;
            cellTO.cellFunctionData.sensor.maxRange = cell->cellFunctionData.sensor.maxRange;
            cellTO.cellFunctionData.sensor.restrictToColor = cell->cellFunctionData.sensor.restrictToColor;
            cellTO.cellFunctionData.sensor.restrictToMutants = cell->cellFunctionData.sensor.restrictToMutants;
            cellTO.cellFunctionData.sensor.memoryChannel1 = cell->cellFunctionData.sensor.memoryChannel1;
            cellTO.cellFunctionData.sensor.memoryChannel2 = cell->cellFunctionData.sensor.memoryChannel2;
            cellTO.cellFunctionData.sensor.memoryChannel3 = cell->cellFunctionData.sensor.memoryChannel3;
            cellTO.cellFunctionData.sensor.memoryTargetX = cell->cellFunctionData.sensor.memoryTargetX;
            cellTO.cellFunctionData.sensor.memoryTargetY = cell->cellFunctionData.sensor.memoryTargetY;
        } break;
        case CellFunction_Nerve: {
            cellTO.cellFunctionData.nerve.pulseMode = cell->cellFunctionData.nerve.pulseMode;
            cellTO.cellFunctionData.nerve.alternationMode = cell->cellFunctionData.nerve.alternationMode;
        } break;
        case CellFunction_Attacker: {
            cellTO.cellFunctionData.attacker.mode = cell->cellFunctionData.attacker.mode;
        } break;
        case CellFunction_Injector: {
            cellTO.cellFunctionData.injector.mode = cell->cellFunctionData.injector.mode;
            cellTO.cellFunctionData.injector.counter = cell->cellFunctionData.injector.counter;
            copyAuxiliaryData(
                cell->cellFunctionData.injector.genomeSize,
                cell->cellFunctionData.injector.genome,
                cellTO.cellFunctionData.injector.genomeSize,
                cellTO.cellFunctionData.injector.genomeDataIndex,
                *dataTO.numAuxiliaryData,
                dataTO.auxiliaryData);
            cellTO.cellFunctionData.injector.genomeGeneration = cell->cellFunctionData.injector.genomeGeneration;
        } break;
        case CellFunction_Muscle: {
            cellTO.cellFunctionData.muscle.mode = cell->cellFunctionData.muscle.mode;
            cellTO.cellFunctionData.muscle.lastBendingDirection = cell->cellFunctionData.muscle.lastBendingDirection;
            cellTO.cellFunctionData.muscle.lastBendingSourceIndex = cell->cellFunctionData.muscle.lastBendingSourceIndex;
            cellTO.cellFunctionData.muscle.consecutiveBendingAngle = cell->cellFunctionData.muscle.consecutiveBendingAngle;
            cellTO.cellFunctionData.muscle.lastMovementX = cell->cellFunctionData.muscle.lastMovementX;
            cellTO.cellFunctionData.muscle.lastMovementY = cell->cellFunctionData.muscle.lastMovementY;
        } break;
        case CellFunction_Defender: {
            cellTO.cellFunctionData.defender.mode = cell->cellFunctionData.defender.mode;
        } break;
        case CellFunction_Reconnector: {
            cellTO.cellFunctionData.reconnector.restrictToColor = cell->cellFunctionData.reconnector.restrictToColor;
            cellTO.cellFunctionData.reconnector.restrictToMutants = cell->cellFunctionData.reconnector.restrictToMutants;
        } break;
        case CellFunction_Detonator: {
            cellTO.cellFunctionData.detonator.state = cell->cellFunctionData.detonator.state;
            cellTO.cellFunctionData.detonator.countdown = cell->cellFunctionData.detonator.countdown;
        } break;
        }
    }

    __device__ void createParticleTO(Particle* particle, DataTO& dataTO)
    {
        int particleTOIndex = alienAtomicAdd64(dataTO.numParticles, uint64_t(1));
        ParticleTO& particleTO = dataTO.particles[particleTOIndex];

        particleTO.id = particle->id;
        particleTO.pos = particle->absPos;
        particleTO.vel = particle->vel;
        particleTO.energy = particle->energy;
        particleTO.color = particle->color;
    }

}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/
__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, DataTO dataTO)
{
    auto const& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.objects.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            cell->tag = -1;
            continue;
        }
        createCellTO(cell, dataTO, cellArrayStart);
    }
}

__global__ void cudaGetSelectedParticleData(SimulationData data, DataTO access)
{
    PartitionData particleBlock = calcPartition(data.objects.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particlePointers.at(particleIndex);
        if (particle->selected == 0) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, DataTO dataTO)
{
    auto const& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.objects.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == 0) {
                break;
            }
            if (ids.values[i] == cell->id) {
                found = true;
            }
        }
        if (!found) {
            cell->tag = -1;
            continue;
        }

        createCellTO(cell, dataTO, cellArrayStart);
    }
}

__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, DataTO access)
{
    PartitionData particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particlePointers.at(particleIndex);
        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == 0) {
                break;
            }
            if (ids.values[i] == particle->id) {
                found = true;
            }
        }
        if (!found) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataTO dataTO)
{
    {
        auto const& cells = data.objects.cellPointers;
        auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);

            if (!Math::isInBetweenModulo(toFloat(rectUpperLeft.x), toFloat(rectLowerRight.x), cell->pos.x, toFloat(data.worldSize.x))) {
                continue;
            }
            if (!Math::isInBetweenModulo(toFloat(rectUpperLeft.y), toFloat(rectLowerRight.y), cell->pos.y, toFloat(data.worldSize.y))) {
                continue;
            }

            auto cellTOIndex = alienAtomicAdd64(dataTO.numCells, uint64_t(1));
            auto& cellTO = dataTO.cells[cellTOIndex];

            cellTO.id = cell->id;
            cellTO.pos = cell->pos;
            cellTO.cellFunction = cell->cellFunction;
            cellTO.selected = cell->selected;
            cellTO.executionOrderNumber = cell->executionOrderNumber;
        }
    }
    {
        auto const& particles = data.objects.particlePointers;
        auto const partition = calcAllThreadsPartition(particles.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);

            auto pos = particle->absPos;
            data.particleMap.correctPosition(pos);
            if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
                continue;
            }
            auto particleTOIndex = alienAtomicAdd64(dataTO.numParticles, uint64_t(1));
            auto& particleTO = dataTO.particles[particleTOIndex];

            particleTO.id = particle->id;
            particleTO.pos = particle->absPos;
            particleTO.selected = particle->selected;
        }
    }
}

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataTO dataTO)
{
    auto const& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.objects.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto pos = cell->pos;
        data.cellMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            cell->tag = -1;
            continue;
        }

        createCellTO(cell, dataTO, cellArrayStart);
    }
}

__global__ void cudaResolveConnections(SimulationData data, DataTO dataTO)
{
    auto const partition = calcAllThreadsPartition(*dataTO.numCells);
    auto const firstCell = data.objects.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cellTO = dataTO.cells[index];

        for (int i = 0; i < cellTO.numConnections; ++i) {
            auto const cellIndex = cellTO.connections[i].cellIndex;
            cellTO.connections[i].cellIndex = data.objects.cells.at(cellIndex).tag;
        }
    }
}

__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataTO access)
{
    PartitionData particleBlock = calcPartition(data.objects.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particlePointers.at(particleIndex);
        auto pos = particle->absPos;
        data.particleMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaCreateDataFromTO(SimulationData data, DataTO dataTO, bool selectNewData, bool createIds)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition = calcPartition(*dataTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto particle = factory.createParticleFromTO(dataTO.particles[index], createIds);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto cellPartition = calcPartition(*dataTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    auto cellTargetArray = data.objects.cells.getArray() + data.objects.cells.getNumOrigEntries();
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto cell = factory.createCellFromTO(dataTO, index, dataTO.cells[index], cellTargetArray, createIds);
        if (selectNewData) {
            cell->selected = 1;
        }
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, DataTO dataTO)
{
    {
        auto const partition = calcPartition(*dataTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = dataTO.cells[index];
            numberGen.adaptMaxId(cell.id);
            numberGen.adaptMaxSmallId(cell.mutationId);
            if (cell.cellFunction == CellFunction_Constructor) {
                numberGen.adaptMaxSmallId(cell.cellFunctionData.constructor.offspringMutationId);
            }
        }
    }
    {
        auto const partition = calcPartition(*dataTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = dataTO.particles[index];
            numberGen.adaptMaxId(particle.id);
        }
    }
}

__global__ void cudaClearDataTO(DataTO dataTO)
{
    *dataTO.numCells = 0;
    *dataTO.numParticles = 0;
    *dataTO.numAuxiliaryData = 0;
}

__global__ void cudaClearData(SimulationData data)
{
    data.objects.cellPointers.reset();
    data.objects.particlePointers.reset();
    data.objects.cells.reset();
    data.objects.particles.reset();
    data.objects.auxiliaryData.reset();
}

__global__ void cudaSaveNumEntries(SimulationData data)
{
    data.objects.saveNumEntries();
}
