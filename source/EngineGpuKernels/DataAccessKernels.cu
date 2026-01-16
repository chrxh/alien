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
                    nodeTO.signalRestriction.mode = node.signalRestriction.mode;
                    nodeTO.signalRestriction.baseAngle = node.signalRestriction.baseAngle;
                    nodeTO.signalRestriction.openingAngle = node.signalRestriction.openingAngle;
                    nodeTO.cellType = node.cellType;
                    switch (node.cellType) {
                    case CellTypeGenome_Base:
                        break;
                    case CellTypeGenome_Depot:
                        nodeTO.cellTypeData.depot.storageLimit = node.cellTypeData.depot.storageLimit;
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
                        nodeTO.cellTypeData.attacker.mode = node.cellTypeData.attacker.mode;
                        if (node.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                            nodeTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor = node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
                        } else if (node.cellTypeData.attacker.mode == AttackerMode_Creature) {
                            nodeTO.cellTypeData.attacker.modeData.attackCreature.minNumCells = node.cellTypeData.attacker.modeData.attackCreature.minNumCells;
                            nodeTO.cellTypeData.attacker.modeData.attackCreature.maxNumCells = node.cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                            nodeTO.cellTypeData.attacker.modeData.attackCreature.restrictToColor = node.cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                            nodeTO.cellTypeData.attacker.modeData.attackCreature.restrictToLineage = node.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
                        }
                        break;
                    case CellTypeGenome_Injector:
                        nodeTO.cellTypeData.injector.geneIndex = node.cellTypeData.injector.geneIndex;
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
                        nodeTO.cellTypeData.reconnector.mode = node.cellTypeData.reconnector.mode;
                        if (node.cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
                        } else if (node.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                            nodeTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor = node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor;
                        } else if (node.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells = node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells = node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor = node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor;
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage = node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
                        }
                        break;
                    case CellTypeGenome_Detonator:
                        nodeTO.cellTypeData.detonator.countdown = node.cellTypeData.detonator.countdown;
                        break;
                    case CellTypeGenome_Digestor:
                        nodeTO.cellTypeData.digestor.rawEnergyConductivity = node.cellTypeData.digestor.rawEnergyConductivity;
                        break;
                    case CellTypeGenome_Memory:
                        nodeTO.cellTypeData.memory.mode = node.cellTypeData.memory.mode;
                        nodeTO.cellTypeData.memory.numSignalEntries = node.cellTypeData.memory.numSignalEntries;
                        nodeTO.cellTypeData.memory.channelBitMask = node.cellTypeData.memory.channelBitMask;
                        if (node.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                            nodeTO.cellTypeData.memory.modeData.signalDelay.delay = node.cellTypeData.memory.modeData.signalDelay.delay;
                        } else if (node.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                            nodeTO.cellTypeData.memory.modeData.signalRecorder.readOnly = node.cellTypeData.memory.modeData.signalRecorder.readOnly;
                            nodeTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries = node.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
                        } else if (node.cellTypeData.memory.mode == MemoryMode_SignalStorage) {
                            nodeTO.cellTypeData.memory.modeData.signalStorage.readOnly = node.cellTypeData.memory.modeData.signalStorage.readOnly;
                        } else if (node.cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
                            nodeTO.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight =
                                node.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
                        }
                        int targetSize;  // not used
                        copyDataToHeap<int>(
                            sizeof(SignalEntryGenome) * node.cellTypeData.memory.numSignalEntries,
                            reinterpret_cast<uint8_t*>(node.cellTypeData.memory.signalEntries),
                            targetSize,
                            nodeTO.cellTypeData.memory.signalEntriesDataIndex,
                            to);
                        break;
                    case CellTypeGenome_Communicator:
                        nodeTO.cellTypeData.communicator.mode = node.cellTypeData.communicator.mode;
                        if (node.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                            nodeTO.cellTypeData.communicator.modeData.sender.range = node.cellTypeData.communicator.modeData.sender.range;
                            nodeTO.cellTypeData.communicator.modeData.sender.maxTimesSent = node.cellTypeData.communicator.modeData.sender.maxTimesSent;
                        } else if (node.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                            nodeTO.cellTypeData.communicator.modeData.receiver.restrictToColor = node.cellTypeData.communicator.modeData.receiver.restrictToColor;
                            nodeTO.cellTypeData.communicator.modeData.receiver.restrictToLineage = node.cellTypeData.communicator.modeData.receiver.restrictToLineage;
                        }
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

    __device__ void createCreatureTO(Object* object, TO& to)
    {
        uint64_t origCreatureTOIndex = alienAtomicExch64(&object->creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origCreatureTOIndex == VALUE_NOT_SET_UINT64) {

            auto creatureTOIndex = alienAtomicAdd64(to.numCreatures, static_cast<uint64_t>(1));
            if (creatureTOIndex >= to.capacities.creatures) {
                printf("Insufficient creature memory for transfer objects.\n");
                ABORT();
            }
            auto& creatureTO = to.creatures[creatureTOIndex];
            auto const& creature = object->creature;
            creatureTO.id = creature->id;
            creatureTO.ancestorId = creature->ancestorId;
            creatureTO.generation = creature->generation;
            creatureTO.lineageId = creature->lineageId;
            creatureTO.numObjects = creature->numObjects;
            creatureTO.frontAngleId = creature->frontAngleId;
            creatureTO.genomeArrayIndex = creature->genome->genomeIndex;

            alienAtomicExch64(&object->creature->creatureIndex, creatureTOIndex);
        } else if (origCreatureTOIndex != 0) {
            alienAtomicExch64(&object->creature->creatureIndex, origCreatureTOIndex);
        }
    }

    __device__ void createObjectTO(Object* object, TO& to, uint8_t* heap)
    {
        auto objectTOIndex = alienAtomicAdd64(to.numObjects, static_cast<uint64_t>(1));
        if (objectTOIndex >= to.capacities.objects) {
            printf("Insufficient cell memory for transfer objects.\n");
            ABORT();
        }
        auto& objectTO = to.objects[objectTOIndex];

        objectTO.id = object->id;
        objectTO.belongToCreature = (object->creature != nullptr);
        if (objectTO.belongToCreature) {
            objectTO.creatureIndex = object->creature->creatureIndex;
        }
        objectTO.pos = object->pos;
        objectTO.vel = object->vel;
        objectTO.fixed = object->fixed;
        objectTO.sticky = object->sticky;
        objectTO.usableEnergy = object->usableEnergy;
        objectTO.rawEnergy = object->rawEnergy;
        objectTO.stiffness = object->stiffness;
        objectTO.numConnections = object->numConnections;
        objectTO.cellState = object->cellState;
        objectTO.cellType = object->cellType;
        objectTO.color = object->color;
        objectTO.frontAngle = object->frontAngle;
        objectTO.age = object->age;
        objectTO.signalRestriction.mode = object->signalRestriction.mode;
        objectTO.signalRestriction.baseAngle = object->signalRestriction.baseAngle;
        objectTO.signalRestriction.openingAngle = object->signalRestriction.openingAngle;
        objectTO.signalState = object->signalState;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            objectTO.signal.channels[i] = object->signal.channels[i];
        }
        objectTO.signal.numTimesSent = object->signal.numTimesSent;
        objectTO.activationTime = object->activationTime;
        objectTO.cellTriggered = object->cellTriggered;
        objectTO.nodeIndex = object->nodeIndex;
        objectTO.parentNodeIndex = object->parentNodeIndex;
        objectTO.geneIndex = object->geneIndex;
        objectTO.frontAngleId = object->frontAngleId;
        objectTO.headCell = object->headCell;

        object->tempValue.as_uint64 = objectTOIndex;
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectingCell = object->connections[i].object;
            objectTO.connections[i].objectIndex = reinterpret_cast<uint8_t*>(connectingCell) - heap;
            objectTO.connections[i].distance = object->connections[i].distance;
            objectTO.connections[i].angleFromPrevious = object->connections[i].angleFromPrevious;
        }

        if (object->neuralNetwork != nullptr) {
            int targetSize;  //not used
            copyDataToHeap<int>(sizeof(NeuralNetwork), reinterpret_cast<uint8_t*>(object->neuralNetwork), targetSize, objectTO.neuralNetworkDataIndex, to);
        } else {
            objectTO.neuralNetworkDataIndex = VALUE_NOT_SET_UINT64;
        }
        switch (object->cellType) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            objectTO.cellTypeData.depot.storageLimit = object->cellTypeData.depot.storageLimit;
            objectTO.cellTypeData.depot.storedUsableEnergy = object->cellTypeData.depot.storedUsableEnergy;
        } break;
        case CellType_Constructor: {
            objectTO.cellTypeData.constructor.autoTriggerInterval = object->cellTypeData.constructor.autoTriggerInterval;
            objectTO.cellTypeData.constructor.constructionActivationTime = object->cellTypeData.constructor.constructionActivationTime;
            objectTO.cellTypeData.constructor.constructionAngle = object->cellTypeData.constructor.constructionAngle;
            objectTO.cellTypeData.constructor.provideEnergy = object->cellTypeData.constructor.provideEnergy;
            objectTO.cellTypeData.constructor.geneIndex = object->cellTypeData.constructor.geneIndex;
            objectTO.cellTypeData.constructor.lastConstructedCellId = object->cellTypeData.constructor.lastConstructedCellId;
            objectTO.cellTypeData.constructor.currentNodeIndex = object->cellTypeData.constructor.currentNodeIndex;
            objectTO.cellTypeData.constructor.currentConcatenation = object->cellTypeData.constructor.currentConcatenation;
            objectTO.cellTypeData.constructor.currentBranch = object->cellTypeData.constructor.currentBranch;
        } break;
        case CellType_Sensor: {
            objectTO.cellTypeData.sensor.autoTriggerInterval = object->cellTypeData.sensor.autoTriggerInterval;
            objectTO.cellTypeData.sensor.minRange = object->cellTypeData.sensor.minRange;
            objectTO.cellTypeData.sensor.maxRange = object->cellTypeData.sensor.maxRange;
            objectTO.cellTypeData.sensor.mode = object->cellTypeData.sensor.mode;
            if (objectTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
            } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                objectTO.cellTypeData.sensor.modeData.detectEnergy.minDensity = object->cellTypeData.sensor.modeData.detectEnergy.minDensity;
            } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
            } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                objectTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity = object->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                objectTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor = object->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
            } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                objectTO.cellTypeData.sensor.modeData.detectCreature.minNumCells = object->cellTypeData.sensor.modeData.detectCreature.minNumCells;
                objectTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells = object->cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                objectTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor = object->cellTypeData.sensor.modeData.detectCreature.restrictToColor;
                objectTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage = object->cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            }
            objectTO.cellTypeData.sensor.lastMatchAvailable = object->cellTypeData.sensor.lastMatchAvailable;
            objectTO.cellTypeData.sensor.lastMatch.creatureId = object->cellTypeData.sensor.lastMatch.creatureId;
            objectTO.cellTypeData.sensor.lastMatch.pos = object->cellTypeData.sensor.lastMatch.pos;
        } break;
        case CellType_Generator: {
            objectTO.cellTypeData.generator.autoTriggerInterval = object->cellTypeData.generator.autoTriggerInterval;
            objectTO.cellTypeData.generator.pulseType = object->cellTypeData.generator.pulseType;
            objectTO.cellTypeData.generator.alternationInterval = object->cellTypeData.generator.alternationInterval;
            objectTO.cellTypeData.generator.numPulses = object->cellTypeData.generator.numPulses;
        } break;
        case CellType_Attacker: {
            objectTO.cellTypeData.attacker.mode = object->cellTypeData.attacker.mode;
            if (object->cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                objectTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor = object->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
            } else if (object->cellTypeData.attacker.mode == AttackerMode_Creature) {
                objectTO.cellTypeData.attacker.modeData.attackCreature.minNumCells = object->cellTypeData.attacker.modeData.attackCreature.minNumCells;
                objectTO.cellTypeData.attacker.modeData.attackCreature.maxNumCells = object->cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                objectTO.cellTypeData.attacker.modeData.attackCreature.restrictToColor = object->cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                objectTO.cellTypeData.attacker.modeData.attackCreature.restrictToLineage = object->cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
            }
        } break;
        case CellType_Injector: {
            objectTO.cellTypeData.injector.geneIndex = object->cellTypeData.injector.geneIndex;
        } break;
        case CellType_Muscle: {
            objectTO.cellTypeData.muscle.mode = object->cellTypeData.muscle.mode;
            if (objectTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
                objectTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = object->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                objectTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = object->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                objectTO.cellTypeData.muscle.modeData.autoBending.initialAngle = object->cellTypeData.muscle.modeData.autoBending.initialAngle;
                objectTO.cellTypeData.muscle.modeData.autoBending.forward = object->cellTypeData.muscle.modeData.autoBending.forward;
                objectTO.cellTypeData.muscle.modeData.autoBending.activation = object->cellTypeData.muscle.modeData.autoBending.activation;
                objectTO.cellTypeData.muscle.modeData.autoBending.activationCountdown = object->cellTypeData.muscle.modeData.autoBending.activationCountdown;
                objectTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = object->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                objectTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = object->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                objectTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio = object->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                objectTO.cellTypeData.muscle.modeData.manualBending.initialAngle = object->cellTypeData.muscle.modeData.manualBending.initialAngle;
                objectTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta = object->cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
                objectTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied =
                    object->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
            } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                objectTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = object->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                objectTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                    object->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                objectTO.cellTypeData.muscle.modeData.angleBending.initialAngle = object->cellTypeData.muscle.modeData.angleBending.initialAngle;
            } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                objectTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = object->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = object->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance = object->cellTypeData.muscle.modeData.autoCrawling.initialDistance;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = object->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.forward = object->cellTypeData.muscle.modeData.autoCrawling.forward;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.activation = object->cellTypeData.muscle.modeData.autoCrawling.activation;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown = object->cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
                objectTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = object->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
            } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                objectTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                    object->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                objectTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                    object->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                objectTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance = object->cellTypeData.muscle.modeData.manualCrawling.initialDistance;
                objectTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = object->cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
                objectTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = object->cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
                objectTO.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied =
                    object->cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
            } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
            }
            objectTO.cellTypeData.muscle.lastMovementX = object->cellTypeData.muscle.lastMovementX;
            objectTO.cellTypeData.muscle.lastMovementY = object->cellTypeData.muscle.lastMovementY;
        } break;
        case CellType_Defender: {
            objectTO.cellTypeData.defender.mode = object->cellTypeData.defender.mode;
        } break;
        case CellType_Reconnector: {
            objectTO.cellTypeData.reconnector.mode = object->cellTypeData.reconnector.mode;
            if (object->cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
            } else if (object->cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                objectTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor = object->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor;
            } else if (object->cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                objectTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells = object->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                objectTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells = object->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                objectTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor = object->cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor;
                objectTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage = object->cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
            }
        } break;
        case CellType_Detonator: {
            objectTO.cellTypeData.detonator.state = object->cellTypeData.detonator.state;
            objectTO.cellTypeData.detonator.countdown = object->cellTypeData.detonator.countdown;
        } break;
        case CellType_Digestor: {
            objectTO.cellTypeData.digestor.rawEnergyConductivity = object->cellTypeData.digestor.rawEnergyConductivity;
        } break;
        case CellType_Memory: {
            objectTO.cellTypeData.memory.mode = object->cellTypeData.memory.mode;
            objectTO.cellTypeData.memory.numSignalEntries = object->cellTypeData.memory.numSignalEntries;
            objectTO.cellTypeData.memory.channelBitMask = object->cellTypeData.memory.channelBitMask;
            if (object->cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                objectTO.cellTypeData.memory.modeData.signalDelay.delay = object->cellTypeData.memory.modeData.signalDelay.delay;
                objectTO.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized = object->cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized;
                objectTO.cellTypeData.memory.modeData.signalDelay.ringBufferIndex = object->cellTypeData.memory.modeData.signalDelay.ringBufferIndex;
            } else if (object->cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                objectTO.cellTypeData.memory.modeData.signalRecorder.readOnly = object->cellTypeData.memory.modeData.signalRecorder.readOnly;
                objectTO.cellTypeData.memory.modeData.signalRecorder.state = object->cellTypeData.memory.modeData.signalRecorder.state;
                objectTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries = object->cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
                objectTO.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries = object->cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries;
            } else if (object->cellTypeData.memory.mode == MemoryMode_SignalStorage) {
                objectTO.cellTypeData.memory.modeData.signalStorage.readOnly = object->cellTypeData.memory.modeData.signalStorage.readOnly;
            } else if (object->cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
                objectTO.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight =
                    object->cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
            }
            int targetSize;  // not used
            copyDataToHeap<int>(
                sizeof(SignalEntry) * object->cellTypeData.memory.numSignalEntries,
                reinterpret_cast<uint8_t*>(object->cellTypeData.memory.signalEntries),
                targetSize,
                objectTO.cellTypeData.memory.signalEntriesDataIndex,
                to);
        } break;
        case CellType_Communicator: {
            objectTO.cellTypeData.communicator.mode = object->cellTypeData.communicator.mode;
            if (object->cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                objectTO.cellTypeData.communicator.modeData.sender.range = object->cellTypeData.communicator.modeData.sender.range;
                objectTO.cellTypeData.communicator.modeData.sender.maxTimesSent = object->cellTypeData.communicator.modeData.sender.maxTimesSent;
            } else if (object->cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                objectTO.cellTypeData.communicator.modeData.receiver.restrictToColor = object->cellTypeData.communicator.modeData.receiver.restrictToColor;
                objectTO.cellTypeData.communicator.modeData.receiver.restrictToLineage = object->cellTypeData.communicator.modeData.receiver.restrictToLineage;
            }
        } break;
        }
    }

    __device__ void createEnergyTO(Energy* particle, TO& to)
    {
        int particleTOIndex = alienAtomicAdd64(to.numEnergyParticles, uint64_t(1));
        if (particleTOIndex >= to.capacities.energyParticles) {
            printf("Insufficient particle memory for transfer objects.\n");
            ABORT();
        }

        EnergyTO& particleTO = to.energyParticles[particleTOIndex];

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
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }
        auto pos = object->pos;
        data.objectMap.correctPosition(pos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            object->creature->creatureIndex = VALUE_NOT_SET_UINT64;
            object->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaPrepareSelectedCreaturesForConversionToTO(bool includeClusters, SimulationData data)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            continue;
        }
        object->creature->creatureIndex = VALUE_NOT_SET_UINT64;
        object->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
    }
}

__global__ void cudaPrepareCreaturesAndGenomesForConversionToTO(InspectedEntityIds ids, SimulationData data)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }
        object->creature->creatureIndex = VALUE_NOT_SET_UINT64;
        object->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
    }
}

__global__ void cudaPrepareCreatureGenomeForConversionToTO(uint64_t creatureId, SimulationData data)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }
        if (object->creature->id == creatureId) {
            object->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaGetSelectedObjectDataWithoutConnections(SimulationData data, bool includeClusters, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    auto const cellArrayStart = data.entities.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            object->tempValue.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }
        createObjectTO(object, to, cellArrayStart);
    }
}

__global__ void cudaGetSelectedEnergyData(SimulationData data, TO access)
{
    auto particleBlock = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; particleIndex += particleBlock.step) {
        auto const& particle = data.entities.energies.at(particleIndex);
        if (particle->selected == 0) {
            continue;
        }

        createEnergyTO(particle, access);
    }
}

__global__ void cudaGetInspectedObjectDataWithoutConnections(InspectedEntityIds ids, SimulationData data, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    auto const heapStart = data.entities.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == Const::MaxInspectedObjects_Break) {
                break;
            }
            if (ids.values[i] == object->id) {
                found = true;
            }
            for (int j = 0; j < object->numConnections; ++j) {
                if (ids.values[i] == object->connections[j].object->id) {
                    found = true;
                }
            }
        }
        if (!found) {
            object->tempValue.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }

        createObjectTO(object, to, heapStart);
    }
}

__global__ void cudaGetInspectedEnergyData(InspectedEntityIds ids, SimulationData data, TO access)
{
    auto particleBlock = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; particleIndex += particleBlock.step) {
        auto const& particle = data.entities.energies.at(particleIndex);
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

        createEnergyTO(particle, access);
    }
}

__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
{
    {
        auto const& objects = data.entities.objects;
        auto const partition = calcSystemThreadPartition(objects.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = objects.at(index);

            if (!Math::isInBetweenModulo(toFloat(rectUpperLeft.x), toFloat(rectLowerRight.x), object->pos.x, toFloat(data.worldSize.x))) {
                continue;
            }
            if (!Math::isInBetweenModulo(toFloat(rectUpperLeft.y), toFloat(rectLowerRight.y), object->pos.y, toFloat(data.worldSize.y))) {
                continue;
            }

            auto objectTOIndex = alienAtomicAdd64(to.numObjects, uint64_t(1));
            auto& objectTO = to.objects[objectTOIndex];

            objectTO.id = object->id;
            objectTO.pos = object->pos;
            objectTO.cellType = object->cellType;
            objectTO.selected = object->selected;
        }
    }
    {
        auto const& particles = data.entities.energies;
        auto const partition = calcSystemThreadPartition(particles.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& particle = particles.at(index);

            auto pos = particle->pos;
            data.energyMap.correctPosition(pos);
            if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
                continue;
            }
            auto particleTOIndex = alienAtomicAdd64(to.numEnergyParticles, uint64_t(1));
            auto& particleTO = to.energyParticles[particleTOIndex];

            particleTO.id = particle->id;
            particleTO.pos = particle->pos;
            particleTO.selected = particle->selected;
        }
    }
}

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        auto pos = object->pos;
        data.objectMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }
        if (!object->creature) {
            continue;
        }
        createGenomeTO(object->creature->genome, to);
    }
}

__global__ void cudaGetSelectedGenomeData(SimulationData data, bool includeClusters, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            continue;
        }
        if (!object->creature) {
            continue;
        }

        createGenomeTO(object->creature->genome, to);
    }
}

__global__ void cudaGetGenomeData(InspectedEntityIds ids, SimulationData data, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }

        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == Const::MaxInspectedObjects_Break) {
                break;
            }
            if (ids.values[i] == object->id) {
                found = true;
            }
            for (int j = 0; j < object->numConnections; ++j) {
                if (ids.values[i] == object->connections[j].object->id) {
                    found = true;
                }
            }
        }
        if (!found) {
            continue;
        }
        createGenomeTO(object->creature->genome, to);
    }
}

__global__ void cudaGetCreatureData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        auto pos = object->pos;
        data.objectMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }
        if (!object->creature) {
            continue;
        }
        createCreatureTO(object, to);
    }
}

__global__ void cudaGetSelectedCreatureData(SimulationData data, bool includeClusters, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            continue;
        }
        if (!object->creature) {
            continue;
        }

        createCreatureTO(object, to);
    }
}

__global__ void cudaGetCreatureData(InspectedEntityIds ids, SimulationData data, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }

        bool found = false;
        for (int i = 0; i < Const::MaxInspectedObjects; ++i) {
            if (ids.values[i] == Const::MaxInspectedObjects_Break) {
                break;
            }
            if (ids.values[i] == object->id) {
                found = true;
            }
            for (int j = 0; j < object->numConnections; ++j) {
                if (ids.values[i] == object->connections[j].object->id) {
                    found = true;
                }
            }
        }
        if (!found) {
            continue;
        }
        createCreatureTO(object, to);
    }
}

__global__ void cudaGetGenomeOfCreature(uint64_t creatureId, SimulationData data, TO to, bool* found)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object->creature) {
            continue;
        }
        if (object->creature->id == creatureId) {
            createGenomeTO(object->creature->genome, to);
            *found = true;
            return;
        }
    }
}

// tags cell with objectTO index and tags objectTO connections with cell index
__global__ void cudaGetObjectDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    auto const heap = data.entities.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        auto pos = object->pos;
        data.objectMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            object->tempValue.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }

        createObjectTO(object, to, heap);
    }
}

__global__ void cudaResolveConnections(SimulationData data, TO to)
{
    auto const partition = calcSystemThreadPartition(*to.numObjects);

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& objectTO = to.objects[index];

        for (int i = 0; i < objectTO.numConnections; ++i) {
            auto const objectIndex = objectTO.connections[i].objectIndex;
            objectTO.connections[i].objectIndex = data.entities.heap.atType<Object>(objectIndex).tempValue.as_uint64;
        }
    }
}

__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO access)
{
    auto particleBlock = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; particleIndex += particleBlock.step) {
        auto const& particle = data.entities.energies.at(particleIndex);
        auto pos = particle->pos;
        data.energyMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }

        createEnergyTO(particle, access);
    }
}

__global__ void cudaGetArraysBasedOnTO(SimulationData data, TO to, Object** cellArray)
{
    *cellArray = data.entities.heap.getTypedSubArray<Object>(*to.numObjects);
}

__global__ void cudaSetGenomeDataFromTO(SimulationData data, TO to)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto partition = calcSystemThreadPartition(*to.numGenomes);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        factory.createGenomeFromTO(to, index);
    }
}

__global__ void cudaSetCreatureDataFromTO(SimulationData data, TO to)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto partition = calcSystemThreadPartition(*to.numCreatures);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        factory.createCreatureFromTO(to, index);
    }
}

__global__ void cudaSetCellAndParticleDataFromTO(SimulationData data, TO to, Object** cellArray, bool selectNewData)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition = calcSystemThreadPartition(*to.numEnergyParticles);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto particle = factory.createParticleFromTO(to.energyParticles[index]);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto cellPartition = calcSystemThreadPartition(*to.numObjects);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto object = factory.createObjectFromTO(to, index, *cellArray);
        if (selectNewData) {
            object->selected = 1;
        }
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, TO to)
{
    Ids maxIds;
    {
        auto const partition = calcSystemThreadPartition(*to.numObjects);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& object = to.objects[index];
            maxIds.entityId = max(maxIds.entityId, object.id);

            if (object.belongToCreature) {
                auto const& creature = to.creatures[object.creatureIndex];
                maxIds.entityId = max(maxIds.entityId, creature.id);
            }
            //maxIds.currentLineageId = max(maxIds.currentLineageId, cell.lineageId);
        }
    }
    {
        auto const partition = calcSystemThreadPartition(*to.numEnergyParticles);

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = to.energyParticles[index];
            maxIds.entityId = max(maxIds.entityId, particle.id);
        }
    }
    numberGen.adaptMaxIds(maxIds);
}

__global__ void cudaClearDataTO(TO to)
{
    *to.numObjects = 0;
    *to.numEnergyParticles = 0;
    *to.numCreatures = 0;
    *to.numGenomes = 0;
    *to.numGenes = 0;
    *to.numNodes = 0;
    *to.heapSize = 0;
}

__global__ void cudaSaveNumEntries(SimulationData data)
{
    data.entities.saveNumEntries();
}

__global__ void cudaClearData(SimulationData data)
{
    data.entities.objects.reset();
    data.entities.energies.reset();
    data.entities.heap.reset();
}

__global__ void cudaEstimateCapacityNeededForTO_step1(SimulationData data)
{
    auto const& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->creature) {
            object->creature->creatureIndex = VALUE_NOT_SET_UINT64;
            object->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaEstimateCapacityNeededForTO_step2(SimulationData data, ArraySizesForTO* arraySizes)
{
    auto const& objects = data.entities.objects;
    auto const& particles = data.entities.energies;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arraySizes->objects = objects.getNumEntries();
        arraySizes->energyParticles = particles.getNumEntries();
    }

    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    uint64_t heapBytes = 0;
    uint64_t numCreatures = 0;
    uint64_t numGenomes = 0;
    uint64_t numGenes = 0;
    uint64_t numNodes = 0;
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->neuralNetwork) {
            heapBytes += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
        }
        if (object->cellType == CellType_Memory) {
            heapBytes += sizeof(SignalEntry) * object->cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
        }
        if (object->creature) {
            auto const& creature = object->creature;
            if (alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(0)) == VALUE_NOT_SET_UINT64) {
                ++numCreatures;
                if (alienAtomicExch64(&creature->genome->genomeIndex, static_cast<uint64_t>(0)) == VALUE_NOT_SET_UINT64) {
                    ++numGenomes;
                    numGenes += creature->genome->numGenes;
                    for (int i = 0, j = creature->genome->numGenes; i < j; ++i) {
                        auto& gene = creature->genome->genes[i];
                        numNodes += gene.numNodes;
                        for (int k = 0; k < gene.numNodes; ++k) {
                            auto& node = gene.nodes[k];
                            if (node.cellType == CellTypeGenome_Memory) {
                                heapBytes += sizeof(SignalEntryGenome) * node.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
                            }
                        }
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
        arraySizes->objectArray = *to.numObjects;
        arraySizes->energyArray = *to.numEnergyParticles;
        alienAtomicAdd64(
            &arraySizes->heap,
            *to.numObjects * (sizeof(Object) + GpuMemoryAlignmentBytes) + *to.numEnergyParticles * (sizeof(Energy) + GpuMemoryAlignmentBytes)
                + *to.numCreatures * (sizeof(Creature) + GpuMemoryAlignmentBytes) + *to.numGenomes * (sizeof(Genome) + GpuMemoryAlignmentBytes)
                + *to.numGenes * (sizeof(Gene) + GpuMemoryAlignmentBytes) + *to.numNodes * (sizeof(Node) + GpuMemoryAlignmentBytes));
    }

    {
        auto partition = calcSystemThreadPartition(*to.numObjects);
        uint64_t heapBytes = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& objectTO = to.objects[index];
            heapBytes += sizeof(Object) + GpuMemoryAlignmentBytes;
            if (objectTO.neuralNetworkDataIndex != VALUE_NOT_SET_UINT64) {
                heapBytes += sizeof(NeuralNetwork) + GpuMemoryAlignmentBytes;
            }
            if (objectTO.cellType == CellType_Memory) {
                heapBytes += sizeof(SignalEntry) * objectTO.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
            }
        }
        alienAtomicAdd64(&arraySizes->heap, heapBytes);
    }

    {
        auto partition = calcSystemThreadPartition(*to.numNodes);
        uint64_t heapBytes = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& nodeTO = to.nodes[index];
            if (nodeTO.cellType == CellTypeGenome_Memory) {
                heapBytes += sizeof(SignalEntryGenome) * nodeTO.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
            }
        }
        alienAtomicAdd64(&arraySizes->heap, heapBytes);
    }
}
