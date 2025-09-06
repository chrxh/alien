#include "DataAccessKernels.cuh"

#include "Base/Macros.h"

namespace
{
    template <typename T>
    __device__ void
    copyDataToHeap(T sourceSize, uint8_t* source, T& targetSize, uint64_t& targetIndex, CollectionTO& collectionTO)
    {
        targetSize = sourceSize;
        if (sourceSize > 0) {
            targetIndex = alienAtomicAdd64(collectionTO.heapSize, static_cast<uint64_t>(sourceSize));
            if (targetIndex + sourceSize > collectionTO.capacities.heap) {
                printf("Insufficient heap memory for transfer objects.\n");
                ABORT();
            }
            for (int i = 0; i < sourceSize; ++i) {
                collectionTO.heap[targetIndex + i] = source[i];
            }
        }
    }

    __device__ void createCreatureTO(Cell* cell, CollectionTO& collectionTO)
    {
        uint64_t origCreatureIndex = alienAtomicExch64(&cell->creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origCreatureIndex == Creature::CreatureIndex_NotSet) {

            auto creatureTOIndex = alienAtomicAdd64(collectionTO.numCreatures, static_cast<uint64_t>(1));
            if (creatureTOIndex >= collectionTO.capacities.creatures) {
                printf("Insufficient genome memory for transfer objects.\n");
                ABORT();
            }
            auto& creatureTO = collectionTO.creatures[creatureTOIndex];
            auto const& creature = cell->creature;
            creatureTO.id = creature->id;
            creatureTO.ancestorId = creature->ancestorId;
            creatureTO.generation = creature->generation;
            creatureTO.lineageId = creature->lineageId;
            creatureTO.numCells = creature->numCells;
            creatureTO.genome.frontAngle = creature->genome.frontAngle;
            creatureTO.genome.numGenes = creature->genome.numGenes;
            for (int i = 0; i < sizeof(creatureTO.genome.name); ++i) {
                creatureTO.genome.name[i] = creature->genome.name[i];
            }

            auto geneTOArrayStartIndex = alienAtomicAdd64(collectionTO.numGenes, static_cast<uint64_t>(creature->genome.numGenes));
            creatureTO.genome.geneArrayIndex = geneTOArrayStartIndex;
            for (int i = 0, j = creature->genome.numGenes; i < j; ++i) {
                auto& geneTO = collectionTO.genes[geneTOArrayStartIndex + i];
                auto const& gene = creature->genome.genes[i];
                geneTO.shape = gene.shape;
                geneTO.separation = gene.separation;
                geneTO.numBranches = gene.numBranches;
                geneTO.angleAlignment = gene.angleAlignment;
                geneTO.stiffness = gene.stiffness;
                geneTO.connectionDistance = gene.connectionDistance;
                geneTO.numConcatenations = gene.numConcatenations;
                geneTO.numNodes = gene.numNodes;
                for (int i = 0; i < sizeof(gene.name); ++i) {
                    geneTO.name[i] = gene.name[i];
                }
                auto nodeTOArrayStartIndex = alienAtomicAdd64(collectionTO.numNodes, static_cast<uint64_t>(gene.numNodes));
                geneTO.nodeArrayIndex = nodeTOArrayStartIndex;
                for (int i = 0, j = gene.numNodes; i < j; ++i) {
                    auto& nodeTO = collectionTO.nodes[nodeTOArrayStartIndex + i];
                    auto const& node = gene.nodes[i];
                    nodeTO.referenceAngle = node.referenceAngle;
                    nodeTO.color = node.color;
                    nodeTO.numAdditionalConnections = node.numAdditionalConnections;
                    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
                        nodeTO.neuralNetwork.weights[i] = node.neuralNetwork.weights[i];
                    }
                    for (int i = 0; i < MAX_CHANNELS; ++i) {
                        nodeTO.neuralNetwork.biases[i] = node.neuralNetwork.biases[i];
                        nodeTO.neuralNetwork.activationFunctions[i] = node.neuralNetwork.activationFunctions[i];
                    }
                    nodeTO.signalRestriction.active = node.signalRestriction.active;
                    nodeTO.signalRestriction.baseAngle = node.signalRestriction.baseAngle;
                    nodeTO.signalRestriction.openingAngle = node.signalRestriction.openingAngle;
                    nodeTO.cellType = node.cellType;
                    switch (node.cellType) {
                    case CellTypeGenome_Base:
                        break;
                    case CellTypeGenome_Depot:
                        nodeTO.cellTypeData.depot.mode = node.cellTypeData.depot.mode;
                        break;
                    case CellTypeGenome_Constructor:
                        nodeTO.cellTypeData.constructor.autoTriggerInterval = node.cellTypeData.constructor.autoTriggerInterval;
                        nodeTO.cellTypeData.constructor.geneIndex = node.cellTypeData.constructor.geneIndex;
                        nodeTO.cellTypeData.constructor.constructionActivationTime = node.cellTypeData.constructor.constructionActivationTime;
                        break;
                    case CellTypeGenome_Sensor:
                        nodeTO.cellTypeData.sensor.autoTriggerInterval = node.cellTypeData.sensor.autoTriggerInterval;
                        nodeTO.cellTypeData.sensor.minDensity = node.cellTypeData.sensor.minDensity;
                        nodeTO.cellTypeData.sensor.minRange = node.cellTypeData.sensor.minRange;
                        nodeTO.cellTypeData.sensor.maxRange = node.cellTypeData.sensor.maxRange;
                        nodeTO.cellTypeData.sensor.restrictToColor = node.cellTypeData.sensor.restrictToColor;
                        nodeTO.cellTypeData.sensor.restrictToCreatures = node.cellTypeData.sensor.restrictToCreatures;
                        break;
                    case CellTypeGenome_Generator:
                        nodeTO.cellTypeData.generator.autoTriggerInterval = node.cellTypeData.generator.autoTriggerInterval;
                        nodeTO.cellTypeData.generator.pulseType = node.cellTypeData.generator.pulseType;
                        nodeTO.cellTypeData.generator.alternationInterval = node.cellTypeData.generator.alternationInterval;
                        break;
                    case CellTypeGenome_Attacker:
                        break;
                    case CellTypeGenome_Injector:
                        nodeTO.cellTypeData.injector.mode = node.cellTypeData.injector.mode;
                        break;
                    case CellTypeGenome_Muscle:
                        nodeTO.cellTypeData.muscle.mode = node.cellTypeData.muscle.mode;
                        switch (nodeTO.cellTypeData.muscle.mode) {
                        case MuscleMode_AutoBending:
                            nodeTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = node.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                            nodeTO.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio = node.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
                            break;
                        case MuscleMode_ManualBending:
                            nodeTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation =
                                node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                            nodeTO.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio =
                                node.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio;
                            break;
                        case MuscleMode_AngleBending:
                            nodeTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation =
                                node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                            nodeTO.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio =
                                node.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio;
                            break;
                        case MuscleMode_AutoCrawling:
                            nodeTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation =
                                node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                            nodeTO.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio =
                                node.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio;
                            break;
                        case MuscleMode_ManualCrawling:
                            nodeTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                                node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                            nodeTO.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio =
                                node.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio;
                            break;
                        case MuscleMode_DirectMovement:
                            break;
                        }
                    case CellTypeGenome_Defender:
                        nodeTO.cellTypeData.defender.mode = node.cellTypeData.defender.mode;
                        break;
                    case CellTypeGenome_Reconnector:
                        nodeTO.cellTypeData.reconnector.restrictToColor = node.cellTypeData.reconnector.restrictToColor;
                        nodeTO.cellTypeData.reconnector.restrictToCreatures = node.cellTypeData.reconnector.restrictToCreatures;
                        break;
                    case CellTypeGenome_Detonator:
                        nodeTO.cellTypeData.detonator.countdown = node.cellTypeData.detonator.countdown;
                        break;
                    }
                }
            }

            alienAtomicExch64(&cell->creature->creatureIndex, creatureTOIndex);
        } else if (origCreatureIndex != 0) {
            alienAtomicExch64(&cell->creature->creatureIndex, origCreatureIndex);
        }
    }

    __device__ void createCellTO(Cell* cell, CollectionTO& collectionTO, uint8_t* heap)
    {
        auto cellTOIndex = alienAtomicAdd64(collectionTO.numCells, 1ull);
        if (cellTOIndex >= collectionTO.capacities.cells) {
            printf("Insufficient cell memory for transfer objects.\n");
            ABORT();
        }
        auto& cellTO = collectionTO.cells[cellTOIndex];

        cellTO.id = cell->id;
        cellTO.belongToCreature = (cell->creature != nullptr);
        if (cellTO.belongToCreature) {
            cellTO.creatureIndex = cell->creature->creatureIndex;
        }
        cellTO.pos = cell->pos;
        cellTO.vel = cell->vel;
        cellTO.barrier = cell->barrier;
        cellTO.sticky = cell->sticky;
        cellTO.energy = cell->energy;
        cellTO.stiffness = cell->stiffness;
        cellTO.numConnections = cell->numConnections;
        cellTO.cellState = cell->cellState;
        cellTO.cellType = cell->cellType;
        cellTO.color = cell->color;
        cellTO.angleToFront = cell->angleToFront;
        cellTO.age = cell->age;
        cellTO.signalRestriction.active = cell->signalRestriction.active;
        cellTO.signalRestriction.baseAngle = cell->signalRestriction.baseAngle;
        cellTO.signalRestriction.openingAngle = cell->signalRestriction.openingAngle;
        cellTO.signalRelaxationTime = cell->signalRelaxationTime;
        cellTO.signal.active = cell->signal.active;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cell->signal.channels[i];
        }
        cellTO.activationTime = cell->activationTime;
        cellTO.detectedByCreatureId = cell->detectedByCreatureId;
        cellTO.cellTriggered = cell->cellTriggered;
        cellTO.nodeIndex = cell->nodeIndex;
        cellTO.parentNodeIndex = cell->parentNodeIndex;
        cellTO.geneIndex = cell->geneIndex;

        cell->tempValue = cellTOIndex;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;
            cellTO.connections[i].cellIndex = reinterpret_cast<uint8_t*>(connectingCell) - heap;
            cellTO.connections[i].distance = cell->connections[i].distance;
            cellTO.connections[i].angleFromPrevious = cell->connections[i].angleFromPrevious;
        }

        if (cell->neuralNetwork != nullptr) {
            int targetSize;  //not used
            copyDataToHeap<int>(
                sizeof(NeuralNetwork),
                reinterpret_cast<uint8_t*>(cell->neuralNetwork),
                targetSize,
                cellTO.neuralNetworkDataIndex,
                collectionTO);
        } else {
            cellTO.neuralNetworkDataIndex = CellTO::NeuralNetworkDataIndex_NotSet;
        }
        switch (cell->cellType) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            cellTO.cellTypeData.depot.mode = cell->cellTypeData.depot.mode;
        } break;
        case CellType_Constructor: {
            cellTO.cellTypeData.constructor.autoTriggerInterval = cell->cellTypeData.constructor.autoTriggerInterval;
            cellTO.cellTypeData.constructor.constructionActivationTime = cell->cellTypeData.constructor.constructionActivationTime;
            cellTO.cellTypeData.constructor.constructionAngle = cell->cellTypeData.constructor.constructionAngle;
            cellTO.cellTypeData.constructor.geneIndex = cell->cellTypeData.constructor.geneIndex;
            cellTO.cellTypeData.constructor.lastConstructedCellId = cell->cellTypeData.constructor.lastConstructedCellId;
            cellTO.cellTypeData.constructor.currentNodeIndex = cell->cellTypeData.constructor.currentNodeIndex;
            cellTO.cellTypeData.constructor.currentConcatenation = cell->cellTypeData.constructor.currentConcatenation;
            cellTO.cellTypeData.constructor.currentBranch = cell->cellTypeData.constructor.currentBranch;
        } break;
        case CellType_Sensor: {
            cellTO.cellTypeData.sensor.autoTriggerInterval = cell->cellTypeData.sensor.autoTriggerInterval;
            cellTO.cellTypeData.sensor.minDensity = cell->cellTypeData.sensor.minDensity;
            cellTO.cellTypeData.sensor.minRange = cell->cellTypeData.sensor.minRange;
            cellTO.cellTypeData.sensor.maxRange = cell->cellTypeData.sensor.maxRange;
            cellTO.cellTypeData.sensor.restrictToColor = cell->cellTypeData.sensor.restrictToColor;
            cellTO.cellTypeData.sensor.restrictToCreatures = cell->cellTypeData.sensor.restrictToCreatures;
        } break;
        case CellType_Generator: {
            cellTO.cellTypeData.generator.autoTriggerInterval = cell->cellTypeData.generator.autoTriggerInterval;
            cellTO.cellTypeData.generator.pulseType = cell->cellTypeData.generator.pulseType;
            cellTO.cellTypeData.generator.alternationInterval = cell->cellTypeData.generator.alternationInterval;
            cellTO.cellTypeData.generator.numPulses = cell->cellTypeData.generator.numPulses;
        } break;
        case CellType_Attacker: {
        } break;
        case CellType_Injector: {
            cellTO.cellTypeData.injector.mode = cell->cellTypeData.injector.mode;
            cellTO.cellTypeData.injector.counter = cell->cellTypeData.injector.counter;
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
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = cell->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio = cell->cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance = cell->cellTypeData.muscle.modeData.autoCrawling.initialDistance;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = cell->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.forward = cell->cellTypeData.muscle.modeData.autoCrawling.forward;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.activation = cell->cellTypeData.muscle.modeData.autoCrawling.activation;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown = cell->cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = cell->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                    cell->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                cellTO.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio = cell->cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio;
                cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance = cell->cellTypeData.muscle.modeData.manualCrawling.initialDistance;
                cellTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = cell->cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
                cellTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = cell->cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
                cellTO.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied =
                    cell->cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
            }
            cellTO.cellTypeData.muscle.lastMovementX = cell->cellTypeData.muscle.lastMovementX;
            cellTO.cellTypeData.muscle.lastMovementY = cell->cellTypeData.muscle.lastMovementY;
        } break;
        case CellType_Defender: {
            cellTO.cellTypeData.defender.mode = cell->cellTypeData.defender.mode;
        } break;
        case CellType_Reconnector: {
            cellTO.cellTypeData.reconnector.restrictToColor = cell->cellTypeData.reconnector.restrictToColor;
            cellTO.cellTypeData.reconnector.restrictToCreatures = cell->cellTypeData.reconnector.restrictToCreatures;
        } break;
        case CellType_Detonator: {
            cellTO.cellTypeData.detonator.state = cell->cellTypeData.detonator.state;
            cellTO.cellTypeData.detonator.countdown = cell->cellTypeData.detonator.countdown;
        } break;
        }
    }

    __device__ void createParticleTO(Particle* particle, CollectionTO& collectionTO)
    {
        int particleTOIndex = alienAtomicAdd64(collectionTO.numParticles, uint64_t(1));
        if (particleTOIndex >= collectionTO.capacities.particles) {
            printf("Insufficient particle memory for transfer objects.\n");
            ABORT();
        }

        ParticleTO& particleTO = collectionTO.particles[particleTOIndex];

        particleTO.id = particle->id;
        particleTO.pos = particle->pos;
        particleTO.vel = particle->vel;
        particleTO.energy = particle->energy;
        particleTO.color = particle->color;
    }

}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/
__global__ void cudaPrepareCreaturesForConversionToTO(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }
        auto pos = cell->pos;
        data.cellMap.correctPosition(pos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            cell->creature->creatureIndex = Creature::CreatureIndex_NotSet;
        }
    }
}

__global__ void cudaPrepareSelectedCreaturesForConversionToTO(bool includeClusters, SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            continue;
        }
        cell->creature->creatureIndex = Creature::CreatureIndex_NotSet;
    }
}

__global__ void cudaPrepareCreaturesForConversionToTO(InspectedEntityIds ids, SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }
        cell->creature->creatureIndex = Creature::CreatureIndex_NotSet;
    }
}

__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.objects.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            cell->tempValue = ConnectionTO::CellIndex_NotSet;
            continue;
        }
        createCellTO(cell, collectionTO, cellArrayStart);
    }
}

__global__ void cudaGetSelectedParticleData(SimulationData data, CollectionTO access)
{
    PartitionData particleBlock = calcPartition(data.objects.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particles.at(particleIndex);
        if (particle->selected == 0) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const heapStart = data.objects.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == Const::MaxInspectedObjects_Break) {
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
            cell->tempValue = ConnectionTO::CellIndex_NotSet;
            continue;
        }

        createCellTO(cell, collectionTO, heapStart);
    }
}

__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, CollectionTO access)
{
    PartitionData particleBlock = calcAllThreadsPartition(data.objects.particles.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particles.at(particleIndex);
        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == Const::MaxInspectedObjects_Break) {
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

__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO)
{
    {
        auto const& cells = data.objects.cells;
        auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);

            if (!Math::isInBetweenModulo(toFloat(rectUpperLeft.x), toFloat(rectLowerRight.x), cell->pos.x, toFloat(data.worldSize.x))) {
                continue;
            }
            if (!Math::isInBetweenModulo(toFloat(rectUpperLeft.y), toFloat(rectLowerRight.y), cell->pos.y, toFloat(data.worldSize.y))) {
                continue;
            }

            auto cellTOIndex = alienAtomicAdd64(collectionTO.numCells, uint64_t(1));
            auto& cellTO = collectionTO.cells[cellTOIndex];

            cellTO.id = cell->id;
            cellTO.pos = cell->pos;
            cellTO.cellType = cell->cellType;
            cellTO.selected = cell->selected;
        }
    }
    {
        auto const& particles = data.objects.particles;
        auto const partition = calcAllThreadsPartition(particles.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);

            auto pos = particle->pos;
            data.particleMap.correctPosition(pos);
            if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
                continue;
            }
            auto particleTOIndex = alienAtomicAdd64(collectionTO.numParticles, uint64_t(1));
            auto& particleTO = collectionTO.particles[particleTOIndex];

            particleTO.id = particle->id;
            particleTO.pos = particle->pos;
            particleTO.selected = particle->selected;
        }
    }
}

__global__ void cudaGetCreatureData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto pos = cell->pos;
        data.cellMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }
        if (!cell->creature) {
            continue;
        }
        createCreatureTO(cell, collectionTO);
    }
}

__global__ void cudaGetSelectedCreatureData(SimulationData data, bool includeClusters, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            continue;
        }
        if (!cell->creature) {
            continue;
        }

        createCreatureTO(cell, collectionTO);
    }
}

__global__ void cudaGetCreatureData(InspectedEntityIds ids, SimulationData data, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }

        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == Const::MaxInspectedObjects_Break) {
                break;
            }
            if (ids.values[i] == cell->id) {
                found = true;
            }
        }
        if (!found) {
            continue;
        }

        createCreatureTO(cell, collectionTO);
    }
}

// tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const heap = data.objects.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto pos = cell->pos;
        data.cellMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            cell->tempValue = ConnectionTO::CellIndex_NotSet;
            continue;
        }

        createCellTO(cell, collectionTO, heap);
    }
}

__global__ void cudaResolveConnections(SimulationData data, CollectionTO collectionTO)
{
    auto const partition = calcAllThreadsPartition(*collectionTO.numCells);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cellTO = collectionTO.cells[index];

        for (int i = 0; i < cellTO.numConnections; ++i) {
            auto const cellIndex = cellTO.connections[i].cellIndex;
            cellTO.connections[i].cellIndex = data.objects.heap.atType<Cell>(cellIndex).tempValue;
        }
    }
}

__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO access)
{
    PartitionData particleBlock = calcPartition(data.objects.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particles.at(particleIndex);
        auto pos = particle->pos;
        data.particleMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaGetArraysBasedOnTO(SimulationData data, CollectionTO collectionTO, Cell** cellArray)
{
    *cellArray = data.objects.heap.getTypedSubArray<Cell>(*collectionTO.numCells);
}

__global__ void cudaSetCreatureDataFromTO(SimulationData data, CollectionTO collectionTO)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto cellPartition = calcAllThreadsPartition(*collectionTO.numCreatures);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        factory.createCreatureFromTO(collectionTO, index);
    }
}

__global__ void cudaSetDataFromTO(SimulationData data, CollectionTO collectionTO, Cell** cellArray, bool selectNewData)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition = calcPartition(*collectionTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto particle = factory.createParticleFromTO(collectionTO.particles[index]);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto cellPartition = calcAllThreadsPartition(*collectionTO.numCells);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto cell = factory.createCellFromTO(collectionTO, index, *cellArray);
        if (selectNewData) {
            cell->selected = 1;
        }
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, CollectionTO collectionTO)
{
    Ids maxIds;
    {
        auto const partition = calcAllThreadsPartition(*collectionTO.numCells);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = collectionTO.cells[index];
            maxIds.currentObjectId = max(maxIds.currentObjectId, cell.id);

            if (cell.belongToCreature) {
                auto const& creature = collectionTO.creatures[cell.creatureIndex];
                maxIds.currentCreatureId = max(maxIds.currentCreatureId, creature.id);
            }
            //maxIds.currentLineageId = max(maxIds.currentLineageId, cell.lineageId);
        }
    }
    {
        auto const partition = calcPartition(*collectionTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = collectionTO.particles[index];
            maxIds.currentObjectId = max(maxIds.currentObjectId, particle.id);
        }
    }
    numberGen.adaptMaxIds(maxIds);
}

__global__ void cudaClearDataTO(CollectionTO collectionTO)
{
    *collectionTO.numCells = 0;
    *collectionTO.numParticles = 0;
    *collectionTO.numCreatures = 0;
    *collectionTO.numGenes = 0;
    *collectionTO.numNodes = 0;
    *collectionTO.heapSize = 0;
}

__global__ void cudaSaveNumEntries(SimulationData data)
{
    data.objects.saveNumEntries();
}

__global__ void cudaClearData(SimulationData data)
{
    data.objects.cells.reset();
    data.objects.particles.reset();
    data.objects.heap.reset();
}

__global__ void cudaEstimateCapacityNeededForTO(SimulationData data, ArraySizesForTO* arraySizes)
{
    auto const& cells = data.objects.cells;
    auto const& particles = data.objects.particles;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arraySizes->cells = cells.getNumEntries();
        arraySizes->particles = particles.getNumEntries();
    }

    auto partition = calcAllThreadsPartition(cells.getNumEntries());
    uint64_t heapBytes = 0;
    uint64_t numGenomes = 0;
    uint64_t numGenes = 0;
    uint64_t numNodes = 0;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->neuralNetwork) {
            heapBytes += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
        }
        if (cell->creature) {
            ++numGenomes;
            auto const& creature = cell->creature;
            numGenes += creature->genome.numGenes;
            for (int i = 0, j = creature->genome.numGenes; i < j; ++i) {
                numNodes += creature->genome.genes[i].numNodes;
            }
        }
    }
    atomicAdd(&arraySizes->creatures, numGenomes);
    atomicAdd(&arraySizes->genes, numGenes);
    atomicAdd(&arraySizes->nodes, numNodes);
    atomicAdd(&arraySizes->heap, heapBytes);
}

__global__ void cudaEstimateCapacityNeededForGpu(CollectionTO collectionTO, ArraySizesForGpu* arraySizes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arraySizes->cellArray = *collectionTO.numCells;
        arraySizes->particleArray = *collectionTO.numParticles;
        atomicAdd(
            &arraySizes->heap,
            *collectionTO.numCells * (sizeof(Cell) + GpuMemoryAlignmentBytes) + *collectionTO.numParticles * (sizeof(Particle) + GpuMemoryAlignmentBytes) 
                + *collectionTO.numCreatures * (sizeof(Creature) + GpuMemoryAlignmentBytes) + *collectionTO.numGenes * (sizeof(Gene) + GpuMemoryAlignmentBytes)
                + *collectionTO.numNodes * (sizeof(Node) + GpuMemoryAlignmentBytes));
    }

    {
        auto partition = calcAllThreadsPartition(*collectionTO.numCells);
        uint64_t heapBytes = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cellTO = collectionTO.cells[index];
            heapBytes += sizeof(Cell) + GpuMemoryAlignmentBytes;
            if (cellTO.neuralNetworkDataIndex != CellTO::NeuralNetworkDataIndex_NotSet) {
                heapBytes += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
            }
        }
        atomicAdd(&arraySizes->heap, heapBytes);
    }
}
