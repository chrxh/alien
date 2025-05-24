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

    __device__ void createGenomeTO(Cell* cell, CollectionTO& collectionTO)
    {
        auto origGenomeIndex = atomicExch(&cell->genome->genomeIndex, 0);  // 0 = member is currently initialized
        if (origGenomeIndex == Genome::GenomeIndex_NotSet) {

            auto genomeTOIndex = atomicAdd(collectionTO.numGenomes, 1ull);
            if (genomeTOIndex >= collectionTO.capacities.genomes) {
                printf("Insufficient genome memory for transfer objects.\n");
                ABORT();
            }
            auto& genomeTO = collectionTO.genomes[genomeTOIndex];
            auto const& genome = cell->genome;
            genomeTO.frontAngle = genome->frontAngle;
            genomeTO.numGenes = genome->numGenes;

            auto geneTOArrayStartIndex = atomicAdd(collectionTO.numGenes, genome->numGenes);
            for (int i = 0, j = genome->numGenes; i < j; ++i) {
                auto& geneTO = collectionTO.genes[geneTOArrayStartIndex + i];
                auto const& gene = genome->genes[i];
                geneTO.shape = gene.shape;
                geneTO.numBranches = gene.numBranches;
                geneTO.separateConstruction = gene.separateConstruction;
                geneTO.angleAlignment = gene.angleAlignment;
                geneTO.stiffness = gene.stiffness;
                geneTO.connectionDistance = gene.connectionDistance;
                geneTO.numRepetitions = gene.numRepetitions;
                geneTO.concatenationAngle1 = gene.concatenationAngle1;
                geneTO.concatenationAngle2 = gene.concatenationAngle2;
                geneTO.numNodes = gene.numNodes;
                auto nodeTOArrayStartIndex = atomicAdd(collectionTO.numNodes, gene.numNodes);
                for (int i = 0, j = gene.numNodes; i < j; ++i) {
                    auto& nodeTO = collectionTO.nodes[nodeTOArrayStartIndex + i];
                    auto const& node = gene.nodes[i];
                    nodeTO.referenceAngle = node.referenceAngle;
                    nodeTO.color = node.color;
                    nodeTO.numRequiredAdditionalConnections = node.numRequiredAdditionalConnections;

                    int targetSize;  //not used
                    copyDataToHeap<int>(
                        sizeof(NeuralNetworkGenome), reinterpret_cast<uint8_t*>(node.neuralNetwork), targetSize, nodeTO.neuralNetworkDataIndex, collectionTO);

                    nodeTO.signalRoutingRestriction.active = node.signalRoutingRestriction.active;
                    nodeTO.signalRoutingRestriction.baseAngle = node.signalRoutingRestriction.baseAngle;
                    nodeTO.signalRoutingRestriction.openingAngle = node.signalRoutingRestriction.openingAngle;
                    nodeTO.cellType = node.cellType;
                    switch (node.cellType) {
                    case CellTypeGenome_Base:
                        break;
                    case CellTypeGenome_Depot:
                        nodeTO.cellTypeData.depot.mode = node.cellTypeData.depot.mode;
                        break;
                    case CellTypeGenome_Constructor:
                        nodeTO.cellTypeData.constructor.autoTriggerInterval = node.cellTypeData.constructor.autoTriggerInterval;
                        nodeTO.cellTypeData.constructor.constructionActivationTime = node.cellTypeData.constructor.constructionActivationTime;
                        nodeTO.cellTypeData.constructor.constructionAngle1 = node.cellTypeData.constructor.constructionAngle1;
                        nodeTO.cellTypeData.constructor.constructionAngle2 = node.cellTypeData.constructor.constructionAngle2;
                        break;
                    case CellTypeGenome_Sensor:
                        nodeTO.cellTypeData.sensor.autoTriggerInterval = node.cellTypeData.sensor.autoTriggerInterval;
                        nodeTO.cellTypeData.sensor.minDensity = node.cellTypeData.sensor.minDensity;
                        nodeTO.cellTypeData.sensor.minRange = node.cellTypeData.sensor.minRange;
                        nodeTO.cellTypeData.sensor.maxRange = node.cellTypeData.sensor.maxRange;
                        nodeTO.cellTypeData.sensor.restrictToColor = node.cellTypeData.sensor.restrictToColor;
                        nodeTO.cellTypeData.sensor.restrictToMutants = node.cellTypeData.sensor.restrictToMutants;
                        break;
                    case CellTypeGenome_Oscillator:
                        nodeTO.cellTypeData.oscillator.autoTriggerInterval = node.cellTypeData.oscillator.autoTriggerInterval;
                        nodeTO.cellTypeData.oscillator.alternationInterval = node.cellTypeData.oscillator.alternationInterval;
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
                        nodeTO.cellTypeData.reconnector.restrictToMutants = node.cellTypeData.reconnector.restrictToMutants;
                        break;
                    case CellTypeGenome_Detonator:
                        nodeTO.cellTypeData.detonator.countdown = node.cellTypeData.detonator.countdown;
                        break;
                    }
                }
            }

            atomicExch(&cell->genome->genomeIndex, genomeTOIndex);
        } else if (origGenomeIndex != 0) {
            atomicExch(&cell->genome->genomeIndex, origGenomeIndex);
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
        cellTO.hasGenome = (cell->genome != nullptr);
        if (cellTO.hasGenome) {
            cellTO.genomeIndex = cell->genome->genomeIndex;
        }
        cellTO.pos = cell->pos;
        cellTO.vel = cell->vel;
        cellTO.barrier = cell->barrier;
        cellTO.sticky = cell->sticky;
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

        copyDataToHeap(
            cell->metadata.nameSize,
            cell->metadata.name,
            cellTO.metadata.nameSize,
            cellTO.metadata.nameDataIndex,
            collectionTO);
        copyDataToHeap(
            cell->metadata.descriptionSize,
            cell->metadata.description,
            cellTO.metadata.descriptionSize,
            cellTO.metadata.descriptionDataIndex,
            collectionTO);

        cell->tempValue = cellTOIndex;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;
            cellTO.connections[i].cellIndex = reinterpret_cast<uint8_t*>(connectingCell) - heap;
            cellTO.connections[i].distance = cell->connections[i].distance;
            cellTO.connections[i].angleFromPrevious = cell->connections[i].angleFromPrevious;
        }

        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            int targetSize;  //not used
            copyDataToHeap<int>(
                sizeof(NeuralNetwork),
                reinterpret_cast<uint8_t*>(cell->neuralNetwork),
                targetSize,
                cellTO.neuralNetworkDataIndex,
                collectionTO);
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
            copyDataToHeap(
                cell->cellTypeData.constructor.genomeSize,
                cell->cellTypeData.constructor.genome,
                cellTO.cellTypeData.constructor.genomeSize,
                cellTO.cellTypeData.constructor.genomeDataIndex,
                collectionTO);
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
        } break;
        case CellType_Injector: {
            cellTO.cellTypeData.injector.mode = cell->cellTypeData.injector.mode;
            cellTO.cellTypeData.injector.counter = cell->cellTypeData.injector.counter;
            copyDataToHeap(
                cell->cellTypeData.injector.genomeSize,
                cell->cellTypeData.injector.genome,
                cellTO.cellTypeData.injector.genomeSize,
                cellTO.cellTypeData.injector.genomeDataIndex,
                collectionTO);
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
            cellTO.cellTypeData.reconnector.restrictToMutants = cell->cellTypeData.reconnector.restrictToMutants;
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
__global__ void cudaPrepareGenomesForConversionToTO(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->genome) {
            continue;
        }
        auto pos = cell->pos;
        data.cellMap.correctPosition(pos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            cell->genome->genomeIndex = Genome::GenomeIndex_NotSet;
        }
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
            cell->tempValue = -1;
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
    auto const cellArrayStart = data.objects.heap.getArray();

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
            cell->tempValue = -1;
            continue;
        }

        createCellTO(cell, collectionTO, cellArrayStart);
    }
}

__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, CollectionTO access)
{
    PartitionData particleBlock = calcAllThreadsPartition(data.objects.particles.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.objects.particles.at(particleIndex);
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

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO)
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
        if (!cell->genome) {
            continue;
        }

        createGenomeTO(cell, collectionTO);
    }
}

__global__ void cudaGetSelectedGenomeData(SimulationData data, bool includeClusters, CollectionTO collectionTO)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            continue;
        }
        if (!cell->genome) {
            continue;
        }

        createGenomeTO(cell, collectionTO);
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
            cell->tempValue = -1;
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

__global__ void cudaSetGenomeDataFromTO(SimulationData data, CollectionTO collectionTO, bool createIds)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto cellPartition = calcAllThreadsPartition(*collectionTO.numGenomes);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        factory.createGenomeFromTO(collectionTO, index, createIds);
    }
}

__global__ void cudaSetDataFromTO(SimulationData data, CollectionTO collectionTO, Cell** cellArray, bool selectNewData, bool createIds)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition = calcPartition(*collectionTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto particle = factory.createParticleFromTO(collectionTO.particles[index], createIds);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto cellPartition = calcAllThreadsPartition(*collectionTO.numCells);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto cell = factory.createCellFromTO(collectionTO, index, *cellArray, createIds);
        if (selectNewData) {
            cell->selected = 1;
        }
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, CollectionTO collectionTO)
{
    {
        auto const partition = calcAllThreadsPartition(*collectionTO.numCells);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = collectionTO.cells[index];
            numberGen.adaptMaxId(cell.id);
            numberGen.adaptMaxSmallId(cell.mutationId);
            if (cell.cellType == CellType_Constructor) {
                numberGen.adaptMaxSmallId(cell.cellTypeData.constructor.offspringMutationId);
            }
        }
    }
    {
        auto const partition = calcPartition(*collectionTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = collectionTO.particles[index];
            numberGen.adaptMaxId(particle.id);
        }
    }
}

__global__ void cudaClearDataTO(CollectionTO collectionTO)
{
    *collectionTO.numCells = 0;
    *collectionTO.numParticles = 0;
    *collectionTO.numGenomes = 0;
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
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        uint64_t dependentDataSize = cell->metadata.nameSize + cell->metadata.descriptionSize + GpuMemoryAlignmentBytes * 2;
        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            dependentDataSize += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
            if (cell->genome) {
                atomicAdd(&arraySizes->genomes, 1ull);
                auto const& genome = cell->genome;
                atomicAdd(&arraySizes->genes, static_cast<uint64_t>(genome->numGenes));
                for (int i = 0, j = genome->numGenes; i < j; ++i) {
                    atomicAdd(&arraySizes->nodes, static_cast<uint64_t>(genome->genes[i].numNodes));
                }
            }
        }
        if (cell->cellType == CellType_Constructor) {
            dependentDataSize += cell->cellTypeData.constructor.genomeSize + GpuMemoryAlignmentBytes;
        } else if (cell->cellType == CellType_Injector) {
            dependentDataSize += cell->cellTypeData.injector.genomeSize + GpuMemoryAlignmentBytes;
        }
        atomicAdd(&arraySizes->heap, dependentDataSize);
    }
}

__global__ void cudaEstimateCapacityNeededForGpu(CollectionTO collectionTO, ArraySizesForGpu* arraySizes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arraySizes->cellArray = *collectionTO.numCells;
        arraySizes->particleArray = *collectionTO.numParticles;
        atomicAdd(
            &arraySizes->heap,
            *collectionTO.numCells * sizeof(Cell) + *collectionTO.numParticles * sizeof(Particle) + *collectionTO.numGenomes * sizeof(Genome)
                + *collectionTO.numGenes * sizeof(Gene) + *collectionTO.numNodes * sizeof(Node) + *collectionTO.numNodes * sizeof(NeuralNetwork)
                + GpuMemoryAlignmentBytes * 6);
    }

    {
        auto partition = calcAllThreadsPartition(*collectionTO.numCells);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = collectionTO.cells[index];
            uint64_t dependentDataSize = sizeof(Cell) + cell.metadata.nameSize + cell.metadata.descriptionSize + GpuMemoryAlignmentBytes * 2;
            if (cell.cellType != CellType_Structure && cell.cellType != CellType_Free) {
                dependentDataSize += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
            }
            if (cell.cellType == CellType_Constructor) {
                dependentDataSize += cell.cellTypeData.constructor.genomeSize + GpuMemoryAlignmentBytes;
            } else if (cell.cellType == CellType_Injector) {
                dependentDataSize += cell.cellTypeData.injector.genomeSize + GpuMemoryAlignmentBytes;
            }
            atomicAdd(&arraySizes->heap, dependentDataSize);
        }
    }
}
