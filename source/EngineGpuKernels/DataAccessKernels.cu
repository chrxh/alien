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
        cellTO.numConnections = cell->numConnections;
        cellTO.livingState = cell->livingState;
        cellTO.creatureId = cell->creatureId;
        cellTO.mutationId = cell->mutationId;
        cellTO.ancestorMutationId = cell->ancestorMutationId;
        cellTO.genomeComplexity = cell->genomeComplexity;
        cellTO.cellType = cell->cellType;
        cellTO.color = cell->color;
        cellTO.angleToFront = cell->angleToFront;
        cellTO.age = cell->age;
        cellTO.signalRoutingRestriction.active = cell->signalRoutingRestriction.active;
        cellTO.signalRoutingRestriction.baseAngle = cell->signalRoutingRestriction.baseAngle;
        cellTO.signalRoutingRestriction.openingAngle = cell->signalRoutingRestriction.openingAngle;
        cellTO.signalRelaxationTime = cell->signalRelaxationTime;
        cellTO.signal.active = cell->signal.active;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cell->signal.channels[i];
        }
        cellTO.activationTime = cell->activationTime;
        cellTO.detectedByCreatureId = cell->detectedByCreatureId;
        cellTO.cellTypeUsed = cell->cellTypeUsed;
        cellTO.genomeNodeIndex = cell->genomeNodeIndex;

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

        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            int targetSize;  //not used
            copyAuxiliaryData<int>(
                sizeof(NeuralNetwork),
                reinterpret_cast<uint8_t*>(cell->neuralNetwork),
                targetSize,
                cellTO.neuralNetwork.dataIndex,
                *dataTO.numAuxiliaryData,
                dataTO.auxiliaryData);
        }
        switch (cell->cellType) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            cellTO.cellTypeData.transmitter.mode = cell->cellTypeData.transmitter.mode;
        } break;
        case CellType_Constructor: {
            cellTO.cellTypeData.constructor.autoTriggerInterval = cell->cellTypeData.constructor.autoTriggerInterval;
            cellTO.cellTypeData.constructor.constructionActivationTime = cell->cellTypeData.constructor.constructionActivationTime;
            copyAuxiliaryData(
                cell->cellTypeData.constructor.genomeSize,
                cell->cellTypeData.constructor.genome,
                cellTO.cellTypeData.constructor.genomeSize,
                cellTO.cellTypeData.constructor.genomeDataIndex,
                *dataTO.numAuxiliaryData,
                dataTO.auxiliaryData);
            cellTO.cellTypeData.constructor.numInheritedGenomeNodes = cell->cellTypeData.constructor.numInheritedGenomeNodes;
            cellTO.cellTypeData.constructor.lastConstructedCellId = cell->cellTypeData.constructor.lastConstructedCellId;
            cellTO.cellTypeData.constructor.genomeCurrentNodeIndex = cell->cellTypeData.constructor.genomeCurrentNodeIndex;
            cellTO.cellTypeData.constructor.genomeCurrentRepetition = cell->cellTypeData.constructor.genomeCurrentRepetition;
            cellTO.cellTypeData.constructor.genomeCurrentBranch = cell->cellTypeData.constructor.genomeCurrentBranch;
            cellTO.cellTypeData.constructor.offspringCreatureId = cell->cellTypeData.constructor.offspringCreatureId;
            cellTO.cellTypeData.constructor.offspringMutationId = cell->cellTypeData.constructor.offspringMutationId;
            cellTO.cellTypeData.constructor.genomeGeneration = cell->cellTypeData.constructor.genomeGeneration;
            cellTO.cellTypeData.constructor.constructionAngle1 = cell->cellTypeData.constructor.constructionAngle1;
            cellTO.cellTypeData.constructor.constructionAngle2 = cell->cellTypeData.constructor.constructionAngle2;
        } break;
        case CellType_Sensor: {
            cellTO.cellTypeData.sensor.autoTriggerInterval = cell->cellTypeData.sensor.autoTriggerInterval;
            cellTO.cellTypeData.sensor.minDensity = cell->cellTypeData.sensor.minDensity;
            cellTO.cellTypeData.sensor.minRange = cell->cellTypeData.sensor.minRange;
            cellTO.cellTypeData.sensor.maxRange = cell->cellTypeData.sensor.maxRange;
            cellTO.cellTypeData.sensor.restrictToColor = cell->cellTypeData.sensor.restrictToColor;
            cellTO.cellTypeData.sensor.restrictToMutants = cell->cellTypeData.sensor.restrictToMutants;
        } break;
        case CellType_Oscillator: {
            cellTO.cellTypeData.oscillator.autoTriggerInterval = cell->cellTypeData.oscillator.autoTriggerInterval;
            cellTO.cellTypeData.oscillator.alternationInterval = cell->cellTypeData.oscillator.alternationInterval;
            cellTO.cellTypeData.oscillator.numPulses = cell->cellTypeData.oscillator.numPulses;
        } break;
        case CellType_Attacker: {
            cellTO.cellTypeData.attacker.mode = cell->cellTypeData.attacker.mode;
        } break;
        case CellType_Injector: {
            cellTO.cellTypeData.injector.mode = cell->cellTypeData.injector.mode;
            cellTO.cellTypeData.injector.counter = cell->cellTypeData.injector.counter;
            copyAuxiliaryData(
                cell->cellTypeData.injector.genomeSize,
                cell->cellTypeData.injector.genome,
                cellTO.cellTypeData.injector.genomeSize,
                cellTO.cellTypeData.injector.genomeDataIndex,
                *dataTO.numAuxiliaryData,
                dataTO.auxiliaryData);
            cellTO.cellTypeData.injector.genomeGeneration = cell->cellTypeData.injector.genomeGeneration;
        } break;
        case CellType_Muscle: {
            cellTO.cellTypeData.muscle.mode = cell->cellTypeData.muscle.mode;
            if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
                cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = cell->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                cellTO.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio = cell->cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
                cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle = cell->cellTypeData.muscle.modeData.autoBending.initialAngle;
                cellTO.cellTypeData.muscle.modeData.autoBending.lastActualAngle = cell->cellTypeData.muscle.modeData.autoBending.lastActualAngle;
                cellTO.cellTypeData.muscle.modeData.autoBending.forward = cell->cellTypeData.muscle.modeData.autoBending.forward;
                cellTO.cellTypeData.muscle.modeData.autoBending.activation = cell->cellTypeData.muscle.modeData.autoBending.activation;
                cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown = cell->cellTypeData.muscle.modeData.autoBending.activationCountdown;
                cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = cell->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = cell->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                cellTO.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio = cell->cellTypeData.muscle.modeData.manualBending.frontBackVelRatio;
                cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle = cell->cellTypeData.muscle.modeData.manualBending.initialAngle;
                cellTO.cellTypeData.muscle.modeData.manualBending.lastActualAngle = cell->cellTypeData.muscle.modeData.manualBending.lastActualAngle;
                cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta = cell->cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
                cellTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied =
                    cell->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = cell->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                cellTO.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio = cell->cellTypeData.muscle.modeData.angleBending.frontBackVelRatio;
                cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle = cell->cellTypeData.muscle.modeData.angleBending.initialAngle;
            }
            cellTO.cellTypeData.muscle.lastMovementX = cell->cellTypeData.muscle.lastMovementX;
            cellTO.cellTypeData.muscle.lastMovementY = cell->cellTypeData.muscle.lastMovementY;
        } break;
        case CellType_Defender: {
            cellTO.cellTypeData.defender.mode = cell->cellTypeData.defender.mode;
        } break;
        case CellType_Reconnector: {
            cellTO.cellTypeData.reconnector.restrictToColor = cell->cellTypeData.reconnector.restrictToColor;
            cellTO.cellTypeData.reconnector.restrictToMutants = cell->cellTypeData.reconnector.restrictToMutants;
        } break;
        case CellType_Detonator: {
            cellTO.cellTypeData.detonator.state = cell->cellTypeData.detonator.state;
            cellTO.cellTypeData.detonator.countdown = cell->cellTypeData.detonator.countdown;
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
            for (int j = 0; j < cell->numConnections; ++j) {
                if (ids.values[i] == cell->connections[j].cell->id) {
                    found = true;
                }
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
            cellTO.cellType = cell->cellType;
            cellTO.selected = cell->selected;
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
            if (cell.cellType == CellType_Constructor) {
                numberGen.adaptMaxSmallId(cell.cellTypeData.constructor.offspringMutationId);
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
