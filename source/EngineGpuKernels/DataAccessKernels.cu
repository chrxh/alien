#include <Base/Macros.h>

#include "DataAccessKernels.cuh"

namespace
{
    template <typename T>
    __device__ void copyDataToHeap(T sourceSize, uint8_t* source, T& targetSize, uint64_t& targetIndex, TO& to)
    {
        targetSize = sourceSize;
        if (sourceSize > 0) {
            targetIndex = alienAtomicAdd64(to.heapSize, static_cast<uint64_t>(sourceSize));
            if (targetIndex + sourceSize > to.capacities.heap) {
                printf("Insufficient heap memory for transfer objects.\n");
                ABORT();
            }
            for (int i = 0; i < sourceSize; ++i) {
                to.heap[targetIndex + i] = source[i];
            }
        }
    }

    __device__ uint64_t createGenomeTO(Genome* genome, TO& to)
    {
        uint64_t origGenomeTOIndex = alienAtomicExch64(&genome->genomeIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origGenomeTOIndex == VALUE_NOT_SET_UINT64) {
            auto genomeTOIndex = alienAtomicAdd64(to.numGenomes, static_cast<uint64_t>(1));
            if (genomeTOIndex >= to.capacities.genomes) {
                printf("Insufficient genome memory for transfer objects.\n");
                ABORT();
            }
            auto& genomeTO = to.genomes[genomeTOIndex];
            genomeTO.id = genome->id;
            genomeTO.frontAngle = genome->frontAngle;
            genomeTO.numGenes = genome->numGenes;
            for (int i = 0; i < sizeof(genomeTO.name); ++i) {
                genomeTO.name[i] = genome->name[i];
            }

            auto geneTOArrayStartIndex = alienAtomicAdd64(to.numGenes, static_cast<uint64_t>(genome->numGenes));
            genomeTO.geneArrayIndex = geneTOArrayStartIndex;
            for (int i = 0, j = genome->numGenes; i < j; ++i) {
                auto& geneTO = to.genes[geneTOArrayStartIndex + i];
                auto const& gene = genome->genes[i];
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
                auto nodeTOArrayStartIndex = alienAtomicAdd64(to.numNodes, static_cast<uint64_t>(gene.numNodes));
                geneTO.nodeArrayIndex = nodeTOArrayStartIndex;
                for (int i = 0, j = gene.numNodes; i < j; ++i) {
                    auto& nodeTO = to.nodes[nodeTOArrayStartIndex + i];
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
                        nodeTO.cellTypeData.constructor.constructionAngle = node.cellTypeData.constructor.constructionAngle;
                        nodeTO.cellTypeData.constructor.provideEnergy = node.cellTypeData.constructor.provideEnergy;
                        break;
                    case CellTypeGenome_Sensor:
                        nodeTO.cellTypeData.sensor.autoTriggerInterval = node.cellTypeData.sensor.autoTriggerInterval;
                        nodeTO.cellTypeData.sensor.minRange = node.cellTypeData.sensor.minRange;
                        nodeTO.cellTypeData.sensor.maxRange = node.cellTypeData.sensor.maxRange;
                        nodeTO.cellTypeData.sensor.mode = node.cellTypeData.sensor.mode;
                        if (nodeTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                            nodeTO.cellTypeData.sensor.modeData.detectEnergy.minDensity = node.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                            nodeTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity = node.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                            nodeTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor = node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.minNumCells = node.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells = node.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor = node.cellTypeData.sensor.modeData.detectCreature.restrictToColor;
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage = node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                        }
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
                            nodeTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio =
                                node.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                            break;
                        case MuscleMode_ManualBending:
                            nodeTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation =
                                node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                            nodeTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio =
                                node.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                            break;
                        case MuscleMode_AngleBending:
                            nodeTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation =
                                node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                            nodeTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                                node.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                            break;
                        case MuscleMode_AutoCrawling:
                            nodeTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation =
                                node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                            nodeTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio =
                                node.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                            break;
                        case MuscleMode_ManualCrawling:
                            nodeTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                                node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                            nodeTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                                node.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
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
                    case CellTypeGenome_Digestor:
                        nodeTO.cellTypeData.digestor.rawEnergyConductivity = node.cellTypeData.digestor.rawEnergyConductivity;
                        break;
                    }
                }
            }

            alienAtomicExch64(&genome->genomeIndex, genomeTOIndex);
            return genomeTOIndex;
        } else if (origGenomeTOIndex != 0) {
            alienAtomicExch64(&genome->genomeIndex, origGenomeTOIndex);
            return origGenomeTOIndex;
        }
        return VALUE_NOT_SET_UINT64;
    }

    __device__ void createCreatureTO(Cell* cell, TO& to)
    {
        uint64_t origCreatureTOIndex = alienAtomicExch64(&cell->creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origCreatureTOIndex == VALUE_NOT_SET_UINT64) {

            auto creatureTOIndex = alienAtomicAdd64(to.numCreatures, static_cast<uint64_t>(1));
            if (creatureTOIndex >= to.capacities.creatures) {
                printf("Insufficient creature memory for transfer objects.\n");
                ABORT();
            }
            auto& creatureTO = to.creatures[creatureTOIndex];
            auto const& creature = cell->creature;
            creatureTO.id = creature->id;
            creatureTO.ancestorId = creature->ancestorId;
            creatureTO.generation = creature->generation;
            creatureTO.lineageId = creature->lineageId;
            creatureTO.numCells = creature->numCells;
            creatureTO.frontAngleId = creature->frontAngleId;
            creatureTO.genomeArrayIndex = creature->genome->genomeIndex;

            alienAtomicExch64(&cell->creature->creatureIndex, creatureTOIndex);
        } else if (origCreatureTOIndex != 0) {
            alienAtomicExch64(&cell->creature->creatureIndex, origCreatureTOIndex);
        }
    }

    __device__ void createCellTO(Cell* cell, TO& to, uint8_t* heap)
    {
        auto cellTOIndex = alienAtomicAdd64(to.numCells, static_cast<uint64_t>(1));
        if (cellTOIndex >= to.capacities.cells) {
            printf("Insufficient cell memory for transfer objects.\n");
            ABORT();
        }
        auto& cellTO = to.cells[cellTOIndex];

        cellTO.id = cell->id;
        cellTO.belongToCreature = (cell->creature != nullptr);
        if (cellTO.belongToCreature) {
            cellTO.creatureIndex = cell->creature->creatureIndex;
        }
        cellTO.pos = cell->pos;
        cellTO.vel = cell->vel;
        cellTO.fixed = cell->fixed;
        cellTO.sticky = cell->sticky;
        cellTO.usableEnergy = cell->usableEnergy;
        cellTO.rawEnergy = cell->rawEnergy;
        cellTO.stiffness = cell->stiffness;
        cellTO.numConnections = cell->numConnections;
        cellTO.cellState = cell->cellState;
        cellTO.cellType = cell->cellType;
        cellTO.color = cell->color;
        cellTO.frontAngle = cell->frontAngle;
        cellTO.age = cell->age;
        cellTO.signalRestriction.active = cell->signalRestriction.active;
        cellTO.signalRestriction.baseAngle = cell->signalRestriction.baseAngle;
        cellTO.signalRestriction.openingAngle = cell->signalRestriction.openingAngle;
        cellTO.signalState = cell->signalState;
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
        cellTO.frontAngleId = cell->frontAngleId;
        cellTO.headCell = cell->headCell;

        cell->tempValue.as_uint64 = cellTOIndex;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;
            cellTO.connections[i].cellIndex = reinterpret_cast<uint8_t*>(connectingCell) - heap;
            cellTO.connections[i].distance = cell->connections[i].distance;
            cellTO.connections[i].angleFromPrevious = cell->connections[i].angleFromPrevious;
        }

        if (cell->neuralNetwork != nullptr) {
            int targetSize;  //not used
            copyDataToHeap<int>(sizeof(NeuralNetwork), reinterpret_cast<uint8_t*>(cell->neuralNetwork), targetSize, cellTO.neuralNetworkDataIndex, to);
        } else {
            cellTO.neuralNetworkDataIndex = VALUE_NOT_SET_UINT64;
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
            cellTO.cellTypeData.constructor.provideEnergy = cell->cellTypeData.constructor.provideEnergy;
            cellTO.cellTypeData.constructor.geneIndex = cell->cellTypeData.constructor.geneIndex;
            cellTO.cellTypeData.constructor.lastConstructedCellId = cell->cellTypeData.constructor.lastConstructedCellId;
            cellTO.cellTypeData.constructor.currentNodeIndex = cell->cellTypeData.constructor.currentNodeIndex;
            cellTO.cellTypeData.constructor.currentConcatenation = cell->cellTypeData.constructor.currentConcatenation;
            cellTO.cellTypeData.constructor.currentBranch = cell->cellTypeData.constructor.currentBranch;
        } break;
        case CellType_Sensor: {
            cellTO.cellTypeData.sensor.autoTriggerInterval = cell->cellTypeData.sensor.autoTriggerInterval;
            cellTO.cellTypeData.sensor.minRange = cell->cellTypeData.sensor.minRange;
            cellTO.cellTypeData.sensor.maxRange = cell->cellTypeData.sensor.maxRange;
            cellTO.cellTypeData.sensor.mode = cell->cellTypeData.sensor.mode;
            if (cellTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                cellTO.cellTypeData.sensor.modeData.detectEnergy.minDensity = cell->cellTypeData.sensor.modeData.detectEnergy.minDensity;
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                cellTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity = cell->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                cellTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor = cell->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                cellTO.cellTypeData.sensor.modeData.detectCreature.minNumCells = cell->cellTypeData.sensor.modeData.detectCreature.minNumCells;
                cellTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells = cell->cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor = cell->cellTypeData.sensor.modeData.detectCreature.restrictToColor;
                cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage = cell->cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            }
            cellTO.cellTypeData.sensor.lastMatchAvailable = cell->cellTypeData.sensor.lastMatchAvailable;
            cellTO.cellTypeData.sensor.lastMatch.creatureId = cell->cellTypeData.sensor.lastMatch.creatureId;
            cellTO.cellTypeData.sensor.lastMatch.pos = cell->cellTypeData.sensor.lastMatch.pos;
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
                cellTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = cell->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle = cell->cellTypeData.muscle.modeData.autoBending.initialAngle;
                cellTO.cellTypeData.muscle.modeData.autoBending.forward = cell->cellTypeData.muscle.modeData.autoBending.forward;
                cellTO.cellTypeData.muscle.modeData.autoBending.activation = cell->cellTypeData.muscle.modeData.autoBending.activation;
                cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown = cell->cellTypeData.muscle.modeData.autoBending.activationCountdown;
                cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = cell->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = cell->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                cellTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio = cell->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle = cell->cellTypeData.muscle.modeData.manualBending.initialAngle;
                cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta = cell->cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
                cellTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied =
                    cell->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = cell->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                cellTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                    cell->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle = cell->cellTypeData.muscle.modeData.angleBending.initialAngle;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = cell->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = cell->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance = cell->cellTypeData.muscle.modeData.autoCrawling.initialDistance;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = cell->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.forward = cell->cellTypeData.muscle.modeData.autoCrawling.forward;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.activation = cell->cellTypeData.muscle.modeData.autoCrawling.activation;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown = cell->cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
                cellTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = cell->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                    cell->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                cellTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                    cell->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
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
        case CellType_Digestor: {
            cellTO.cellTypeData.digestor.rawEnergyConductivity = cell->cellTypeData.digestor.rawEnergyConductivity;
        } break;
        }
    }

    __device__ void createParticleTO(Particle* particle, TO& to)
    {
        int particleTOIndex = alienAtomicAdd64(to.numParticles, uint64_t(1));
        if (particleTOIndex >= to.capacities.particles) {
            printf("Insufficient particle memory for transfer objects.\n");
            ABORT();
        }

        ParticleTO& particleTO = to.particles[particleTOIndex];

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
__global__ void cudaPrepareCreaturesAndGenomesForConversionToTO(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
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
            cell->creature->creatureIndex = VALUE_NOT_SET_UINT64;
            cell->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
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
        cell->creature->creatureIndex = VALUE_NOT_SET_UINT64;
        cell->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
    }
}

__global__ void cudaPrepareCreaturesAndGenomesForConversionToTO(InspectedEntityIds ids, SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }
        cell->creature->creatureIndex = VALUE_NOT_SET_UINT64;
        cell->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
    }
}

__global__ void cudaPrepareCreatureGenomeForConversionToTO(uint64_t creatureId, SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }
        if (cell->creature->id == creatureId) {
            cell->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, TO to)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.objects.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            cell->tempValue.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }
        createCellTO(cell, to, cellArrayStart);
    }
}

__global__ void cudaGetSelectedParticleData(SimulationData data, TO access)
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

__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, TO to)
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
            cell->tempValue.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }

        createCellTO(cell, to, heapStart);
    }
}

__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, TO access)
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

__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
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

            auto cellTOIndex = alienAtomicAdd64(to.numCells, uint64_t(1));
            auto& cellTO = to.cells[cellTOIndex];

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
            auto particleTOIndex = alienAtomicAdd64(to.numParticles, uint64_t(1));
            auto& particleTO = to.particles[particleTOIndex];

            particleTO.id = particle->id;
            particleTO.pos = particle->pos;
            particleTO.selected = particle->selected;
        }
    }
}

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
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
        createGenomeTO(cell->creature->genome, to);
    }
}

__global__ void cudaGetSelectedGenomeData(SimulationData data, bool includeClusters, TO to)
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

        createGenomeTO(cell->creature->genome, to);
    }
}

__global__ void cudaGetGenomeData(InspectedEntityIds ids, SimulationData data, TO to)
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
            for (int j = 0; j < cell->numConnections; ++j) {
                if (ids.values[i] == cell->connections[j].cell->id) {
                    found = true;
                }
            }
        }
        if (!found) {
            continue;
        }
        createGenomeTO(cell->creature->genome, to);
    }
}

__global__ void cudaGetCreatureData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
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
        createCreatureTO(cell, to);
    }
}

__global__ void cudaGetSelectedCreatureData(SimulationData data, bool includeClusters, TO to)
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

        createCreatureTO(cell, to);
    }
}

__global__ void cudaGetCreatureData(InspectedEntityIds ids, SimulationData data, TO to)
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
            for (int j = 0; j < cell->numConnections; ++j) {
                if (ids.values[i] == cell->connections[j].cell->id) {
                    found = true;
                }
            }
        }
        if (!found) {
            continue;
        }
        createCreatureTO(cell, to);
    }
}

__global__ void cudaGetGenomeOfCreature(uint64_t creatureId, SimulationData data, TO to, bool* found)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (!cell->creature) {
            continue;
        }
        if (cell->creature->id == creatureId) {
            createGenomeTO(cell->creature->genome, to);
            *found = true;
            return;
        }
    }
}

// tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
{
    auto const& cells = data.objects.cells;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const heap = data.objects.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto pos = cell->pos;
        data.cellMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            cell->tempValue.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }

        createCellTO(cell, to, heap);
    }
}

__global__ void cudaResolveConnections(SimulationData data, TO to)
{
    auto const partition = calcAllThreadsPartition(*to.numCells);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cellTO = to.cells[index];

        for (int i = 0; i < cellTO.numConnections; ++i) {
            auto const cellIndex = cellTO.connections[i].cellIndex;
            cellTO.connections[i].cellIndex = data.objects.heap.atType<Cell>(cellIndex).tempValue.as_uint64;
        }
    }
}

__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO access)
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

__global__ void cudaGetArraysBasedOnTO(SimulationData data, TO to, Cell** cellArray)
{
    *cellArray = data.objects.heap.getTypedSubArray<Cell>(*to.numCells);
}

__global__ void cudaSetGenomeDataFromTO(SimulationData data, TO to)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto partition = calcAllThreadsPartition(*to.numGenomes);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        factory.createGenomeFromTO(to, index);
    }
}

__global__ void cudaSetCreatureDataFromTO(SimulationData data, TO to)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto partition = calcAllThreadsPartition(*to.numCreatures);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        factory.createCreatureFromTO(to, index);
    }
}

__global__ void cudaSetCellAndParticleDataFromTO(SimulationData data, TO to, Cell** cellArray, bool selectNewData)
{
    __shared__ ObjectFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition = calcPartition(*to.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto particle = factory.createParticleFromTO(to.particles[index]);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto cellPartition = calcAllThreadsPartition(*to.numCells);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto cell = factory.createCellFromTO(to, index, *cellArray);
        if (selectNewData) {
            cell->selected = 1;
        }
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, TO to)
{
    Ids maxIds;
    {
        auto const partition = calcAllThreadsPartition(*to.numCells);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = to.cells[index];
            maxIds.entityId = max(maxIds.entityId, cell.id);

            if (cell.belongToCreature) {
                auto const& creature = to.creatures[cell.creatureIndex];
                maxIds.entityId = max(maxIds.entityId, creature.id);
            }
            //maxIds.currentLineageId = max(maxIds.currentLineageId, cell.lineageId);
        }
    }
    {
        auto const partition = calcPartition(*to.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = to.particles[index];
            maxIds.entityId = max(maxIds.entityId, particle.id);
        }
    }
    numberGen.adaptMaxIds(maxIds);
}

__global__ void cudaClearDataTO(TO to)
{
    *to.numCells = 0;
    *to.numParticles = 0;
    *to.numCreatures = 0;
    *to.numGenomes = 0;
    *to.numGenes = 0;
    *to.numNodes = 0;
    *to.heapSize = 0;
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

__global__ void cudaEstimateCapacityNeededForTO_step1(SimulationData data)
{
    auto const& cells = data.objects.cells;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->creature) {
            cell->creature->creatureIndex = VALUE_NOT_SET_UINT64;
            cell->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaEstimateCapacityNeededForTO_step2(SimulationData data, ArraySizesForTO* arraySizes)
{
    auto const& cells = data.objects.cells;
    auto const& particles = data.objects.particles;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arraySizes->cells = cells.getNumEntries();
        arraySizes->particles = particles.getNumEntries();
    }

    auto partition = calcAllThreadsPartition(cells.getNumEntries());
    uint64_t heapBytes = 0;
    uint64_t numCreatures = 0;
    uint64_t numGenomes = 0;
    uint64_t numGenes = 0;
    uint64_t numNodes = 0;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->neuralNetwork) {
            heapBytes += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
        }
        if (cell->creature) {
            auto const& creature = cell->creature;
            if (alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(0)) == VALUE_NOT_SET_UINT64) {
                ++numCreatures;
                if (alienAtomicExch64(&creature->genome->genomeIndex, static_cast<uint64_t>(0)) == VALUE_NOT_SET_UINT64) {
                    ++numGenomes;
                    numGenes += creature->genome->numGenes;
                    for (int i = 0, j = creature->genome->numGenes; i < j; ++i) {
                        numNodes += creature->genome->genes[i].numNodes;
                    }
                }
            }
        }
    }
    alienAtomicAdd64(&arraySizes->creatures, numCreatures);
    alienAtomicAdd64(&arraySizes->genomes, numGenomes);
    alienAtomicAdd64(&arraySizes->genes, numGenes);
    alienAtomicAdd64(&arraySizes->nodes, numNodes);
    alienAtomicAdd64(&arraySizes->heap, heapBytes);
}

__global__ void cudaEstimateCapacityNeededForGpu(TO to, ArraySizesForGpu* arraySizes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arraySizes->cellArray = *to.numCells;
        arraySizes->particleArray = *to.numParticles;
        alienAtomicAdd64(
            &arraySizes->heap,
            *to.numCells * (sizeof(Cell) + GpuMemoryAlignmentBytes) + *to.numParticles * (sizeof(Particle) + GpuMemoryAlignmentBytes)
                + *to.numCreatures * (sizeof(Creature) + GpuMemoryAlignmentBytes) + *to.numGenomes * (sizeof(Genome) + GpuMemoryAlignmentBytes)
                + *to.numGenes * (sizeof(Gene) + GpuMemoryAlignmentBytes) + *to.numNodes * (sizeof(Node) + GpuMemoryAlignmentBytes));
    }

    {
        auto partition = calcAllThreadsPartition(*to.numCells);
        uint64_t heapBytes = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cellTO = to.cells[index];
            heapBytes += sizeof(Cell) + GpuMemoryAlignmentBytes;
            if (cellTO.neuralNetworkDataIndex != VALUE_NOT_SET_UINT64) {
                heapBytes += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
            }
        }
        alienAtomicAdd64(&arraySizes->heap, heapBytes);
    }
}
