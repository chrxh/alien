#include <Base/Macros.h>

#include "DataAccessKernels.cuh"

namespace
{
    template <typename T>
    __device__ void copyDataToHeap(T sourceSize, uint8_t* source, T& targetSize, uint64_t& targetIndex, TOs& to)
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

    __device__ uint64_t createGenomeTO(Genome* genome, TOs& to)
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
            genomeTO.lineageId = genome->lineageId;
            genomeTO.prevLineageId = genome->prevLineageId;
            genomeTO.frontAngle = genome->frontAngle;
            genomeTO.mutationRates.accumulatedMutations = genome->mutationRates.accumulatedMutations;
            genomeTO.mutationRates.neuronMutation1 = {
                genome->mutationRates.neuronMutation1.probability,
                genome->mutationRates.neuronMutation1.weightSigma,
                genome->mutationRates.neuronMutation1.biasSigma,
                genome->mutationRates.neuronMutation1.activationFunctionProbability};
            genomeTO.mutationRates.neuronMutation2 = {
                genome->mutationRates.neuronMutation2.probability,
                genome->mutationRates.neuronMutation2.weightSigma,
                genome->mutationRates.neuronMutation2.biasSigma,
                genome->mutationRates.neuronMutation2.activationFunctionProbability};
            genomeTO.mutationRates.connectionMutation1 = {
                genome->mutationRates.connectionMutation1.probability, genome->mutationRates.connectionMutation1.sigma};
            genomeTO.mutationRates.connectionMutation2 = {
                genome->mutationRates.connectionMutation2.probability, genome->mutationRates.connectionMutation2.sigma};
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
                geneTO.stiffness = gene.stiffness;
                geneTO.connectionDistance = gene.connectionDistance;
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
                    for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
                        nodeTO.neuralNetwork.weights[i] = node.neuralNetwork.weights[i];
                    }
                    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                        nodeTO.neuralNetwork.biases[i] = node.neuralNetwork.biases[i];
                        nodeTO.neuralNetwork.activationFunctions[i] = node.neuralNetwork.activationFunctions[i];
                    }
                    for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
                        nodeTO.neuralNetwork.connectionWeights[i] = node.neuralNetwork.connectionWeights[i];
                    }
                    nodeTO.cellType = node.cellType;
                    switch (node.cellType) {
                    case CellType_Base:
                        break;
                    case CellType_Depot:
                        nodeTO.cellTypeData.depot.storageLimit = node.cellTypeData.depot.storageLimit;
                        nodeTO.cellTypeData.depot.initialStoredUsableEnergy = node.cellTypeData.depot.initialStoredUsableEnergy;
                        break;
                    case CellType_Sensor:
                        nodeTO.cellTypeData.sensor.autoTrigger = node.cellTypeData.sensor.autoTrigger;
                        nodeTO.cellTypeData.sensor.tagForAttackers = node.cellTypeData.sensor.tagForAttackers;
                        nodeTO.cellTypeData.sensor.minRange = node.cellTypeData.sensor.minRange;
                        nodeTO.cellTypeData.sensor.maxRange = node.cellTypeData.sensor.maxRange;
                        nodeTO.cellTypeData.sensor.mode = node.cellTypeData.sensor.mode;
                        if (nodeTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                            nodeTO.cellTypeData.sensor.modeData.detectEnergy.minDensity = node.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectSolid) {
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                            nodeTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity = node.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                            nodeTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors =
                                node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors;
                        } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.minNumCells = node.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells = node.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToColors =
                                node.cellTypeData.sensor.modeData.detectCreature.restrictToColors;
                            nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage =
                                node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                        }
                        break;
                    case CellType_Generator:
                        nodeTO.cellTypeData.generator.additive = node.cellTypeData.generator.additive;
                        nodeTO.cellTypeData.generator.minValue = node.cellTypeData.generator.minValue;
                        nodeTO.cellTypeData.generator.maxValue = node.cellTypeData.generator.maxValue;
                        nodeTO.cellTypeData.generator.timeOffset = node.cellTypeData.generator.timeOffset;
                        nodeTO.cellTypeData.generator.mode = node.cellTypeData.generator.mode;
                        if (node.cellTypeData.generator.mode == GeneratorMode_SquareSignal) {
                            nodeTO.cellTypeData.generator.modeData.squareSignal.period = node.cellTypeData.generator.modeData.squareSignal.period;
                        } else if (node.cellTypeData.generator.mode == GeneratorMode_SawtoothSignal) {
                            nodeTO.cellTypeData.generator.modeData.sawtoothSignal.period = node.cellTypeData.generator.modeData.sawtoothSignal.period;
                        }
                        break;
                    case CellType_Attacker:
                        nodeTO.cellTypeData.attacker.mode = node.cellTypeData.attacker.mode;
                        if (node.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                            nodeTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors =
                                node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors;
                        }
                        break;
                    case CellType_Injector:
                        nodeTO.cellTypeData.injector.geneIndex = node.cellTypeData.injector.geneIndex;
                        break;
                    case CellType_Muscle:
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
                        break;
                    case CellType_Defender:
                        nodeTO.cellTypeData.defender.mode = node.cellTypeData.defender.mode;
                        break;
                    case CellType_Reconnector:
                        nodeTO.cellTypeData.reconnector.mode = node.cellTypeData.reconnector.mode;
                        if (node.cellTypeData.reconnector.mode == ReconnectorMode_Solid) {
                        } else if (node.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                            nodeTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors =
                                node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors;
                        } else if (node.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells =
                                node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells =
                                node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors =
                                node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors;
                            nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage =
                                node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
                        }
                        break;
                    case CellType_Detonator:
                        nodeTO.cellTypeData.detonator.countdown = node.cellTypeData.detonator.countdown;
                        break;
                    case CellType_Digestor:
                        nodeTO.cellTypeData.digestor.rawEnergyConductivity = node.cellTypeData.digestor.rawEnergyConductivity;
                        break;
                    case CellType_Memory:
                        nodeTO.cellTypeData.memory.mode = node.cellTypeData.memory.mode;
                        nodeTO.cellTypeData.memory.numSignalEntries = node.cellTypeData.memory.numSignalEntries;
                        nodeTO.cellTypeData.memory.channelBitMask = node.cellTypeData.memory.channelBitMask;
                        if (node.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                            nodeTO.cellTypeData.memory.modeData.signalDelay.delay = node.cellTypeData.memory.modeData.signalDelay.delay;
                        } else if (node.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                            nodeTO.cellTypeData.memory.modeData.signalRecorder.readOnly = node.cellTypeData.memory.modeData.signalRecorder.readOnly;
                            nodeTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries =
                                node.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
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
                    case CellType_Communicator:
                        nodeTO.cellTypeData.communicator.mode = node.cellTypeData.communicator.mode;
                        if (node.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                            nodeTO.cellTypeData.communicator.modeData.sender.range = node.cellTypeData.communicator.modeData.sender.range;
                            nodeTO.cellTypeData.communicator.modeData.sender.maxTimesSent = node.cellTypeData.communicator.modeData.sender.maxTimesSent;
                        } else if (node.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                            nodeTO.cellTypeData.communicator.modeData.receiver.restrictToColors =
                                node.cellTypeData.communicator.modeData.receiver.restrictToColors;
                            nodeTO.cellTypeData.communicator.modeData.receiver.restrictToLineage =
                                node.cellTypeData.communicator.modeData.receiver.restrictToLineage;
                        }
                        break;
                    }

                    // Handle optional constructor field
                    nodeTO.constructorAvailable = node.constructorAvailable;
                    if (node.constructorAvailable) {
                        nodeTO.constructor.autoTriggerInterval = node.constructor.autoTriggerInterval;
                        nodeTO.constructor.geneIndex = node.constructor.geneIndex;
                        nodeTO.constructor.constructionActivationTime = node.constructor.constructionActivationTime;
                        nodeTO.constructor.constructionAngle = node.constructor.constructionAngle;
                        nodeTO.constructor.provideEnergy = node.constructor.provideEnergy;
                        nodeTO.constructor.reservedEnergy = node.constructor.reservedEnergy;
                        nodeTO.constructor.separation = node.constructor.separation;
                        nodeTO.constructor.numBranches = node.constructor.numBranches;
                        nodeTO.constructor.numConcatenations = node.constructor.numConcatenations;
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

    __device__ void createCreatureTO(Object* object, TOs& to)
    {
        uint64_t origCreatureTOIndex =
            alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origCreatureTOIndex == VALUE_NOT_SET_UINT64) {

            auto creatureTOIndex = alienAtomicAdd64(to.numCreatures, static_cast<uint64_t>(1));
            if (creatureTOIndex >= to.capacities.creatures) {
                printf("Insufficient creature memory for transfer objects.\n");
                ABORT();
            }
            auto& creatureTO = to.creatures[creatureTOIndex];
            auto const& creature = object->typeData.cell.creature;
            creatureTO.id = creature->id;
            creatureTO.ancestorId = creature->ancestorId;
            creatureTO.generation = creature->generation;
            creatureTO.numCells = creature->numCells;
            creatureTO.mutationState = creature->mutationState;
            creatureTO.headUpdateId = creature->headUpdateId;
            creatureTO.genomeArrayIndex = creature->genome->genomeIndex;

            alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, creatureTOIndex);
        } else if (origCreatureTOIndex != 0) {
            alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, origCreatureTOIndex);
        }
    }

    __device__ void createObjectTO(Object* object, TOs& to, uint8_t* heap)
    {
        auto objectTOIndex = alienAtomicAdd64(to.numObjects, static_cast<uint64_t>(1));
        if (objectTOIndex >= to.capacities.objects) {
            printf("Insufficient cell memory for transfer objects.\n");
            ABORT();
        }
        auto& objectTO = to.objects[objectTOIndex];

        objectTO.id = object->id;
        objectTO.numConnections = object->numConnections;
        objectTO.pos = object->pos;
        objectTO.vel = object->vel;
        objectTO.stiffness = object->stiffness;
        objectTO.color = object->color;
        objectTO.fixed = object->fixed;
        objectTO.sticky = object->sticky;
        objectTO.type = object->type;

        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            objectTO.connections[i].objectIndex = reinterpret_cast<uint8_t*>(connectedObject) - heap;
            objectTO.connections[i].distance = object->connections[i].distance;
            objectTO.connections[i].angleFromPrevious = object->connections[i].angleFromPrevious;
        }

        object->tempValue1.as_uint64 = objectTOIndex;

        if (object->type == ObjectType_Solid) {
            objectTO.typeData.solid.energy = object->typeData.solid.energy;
        } else if (object->type == ObjectType_Fluid) {
            objectTO.typeData.fluid.energy = object->typeData.fluid.energy;
            objectTO.typeData.fluid.glow = object->typeData.fluid.glow;
        } else if (object->type == ObjectType_FreeCell) {
            objectTO.typeData.freeCell.energy = object->typeData.freeCell.energy;
            objectTO.typeData.freeCell.age = object->typeData.freeCell.age;
        } else if (object->type == ObjectType_Cell) {
            auto& cellTO = objectTO.typeData.cell;
            auto const& cell = object->typeData.cell;

            cellTO.creatureIndex = cell.creature->creatureIndex;
            cellTO.usableEnergy = cell.usableEnergy;
            cellTO.rawEnergy = cell.rawEnergy;
            cellTO.cellState = cell.cellState;
            cellTO.frontAngle = cell.frontAngle;
            cellTO.age = cell.age;
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                cellTO.signal.channels[i] = cell.signal.channels[i];
            }
            cellTO.signal.numTimesSent = cell.signal.numTimesSent;
            cellTO.activationTime = cell.activationTime;
            cellTO.lastUpdate = cell.lastUpdate;
            cellTO.concatenationIndex = cell.concatenationIndex;
            cellTO.branchIndex = cell.branchIndex;
            cellTO.nodeIndex = cell.nodeIndex;
            cellTO.parentNodeIndex = cell.parentNodeIndex;
            cellTO.geneIndex = cell.geneIndex;
            cellTO.headUpdateId = cell.headUpdateId;
            cellTO.headCell = cell.headCell;
            cellTO.event = cell.event;
            cellTO.eventCounter = cell.eventCounter;
            cellTO.eventPos = cell.eventPos;

            // Copy NeuralNet to NeuralNetTO
            if (cell.neuralNetwork != nullptr) {
                uint64_t size = sizeof(NeuralNetTO);
                cellTO.neuralNetworkDataIndex = alienAtomicAdd64(to.heapSize, size);
                if (cellTO.neuralNetworkDataIndex + size > to.capacities.heap) {
                    printf("Insufficient heap memory for NeuralNetTO.\n");
                    ABORT();
                }
                auto* nnTO = reinterpret_cast<NeuralNetTO*>(&to.heap[cellTO.neuralNetworkDataIndex]);
                for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
                    nnTO->weights[i] = cell.neuralNetwork->weights[i];
                }
                for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                    nnTO->biases[i] = cell.neuralNetwork->biases[i];
                    nnTO->activationFunctions[i] = cell.neuralNetwork->activationFunctions[i];
                }
                for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
                    nnTO->connectionWeights[i] = cell.neuralNetwork->connectionWeights[i];
                }
            }

            cellTO.cellType = cell.cellType;
            switch (cell.cellType) {
            case CellType_Base: {
            } break;
            case CellType_Depot: {
                cellTO.cellTypeData.depot.storageLimit = cell.cellTypeData.depot.storageLimit;
                cellTO.cellTypeData.depot.storedUsableEnergy = cell.cellTypeData.depot.storedUsableEnergy;
            } break;
            case CellType_Sensor: {
                cellTO.cellTypeData.sensor.autoTrigger = cell.cellTypeData.sensor.autoTrigger;
                cellTO.cellTypeData.sensor.tagForAttackers = cell.cellTypeData.sensor.tagForAttackers;
                cellTO.cellTypeData.sensor.minRange = cell.cellTypeData.sensor.minRange;
                cellTO.cellTypeData.sensor.maxRange = cell.cellTypeData.sensor.maxRange;
                cellTO.cellTypeData.sensor.mode = cell.cellTypeData.sensor.mode;
                if (cellTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                    cellTO.cellTypeData.sensor.modeData.detectEnergy.minDensity = cell.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectSolid) {
                } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                    cellTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity = cell.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                    cellTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors = cell.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors;
                } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                    cellTO.cellTypeData.sensor.modeData.detectCreature.minNumCells = cell.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                    cellTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells = cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                    cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToColors = cell.cellTypeData.sensor.modeData.detectCreature.restrictToColors;
                    cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage = cell.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                }
                cellTO.cellTypeData.sensor.lastMatchAvailable = cell.cellTypeData.sensor.lastMatchAvailable;
                cellTO.cellTypeData.sensor.lastMatch.creatureIdPart = cell.cellTypeData.sensor.lastMatch.creatureIdPart;
                cellTO.cellTypeData.sensor.lastMatch.pos = cell.cellTypeData.sensor.lastMatch.pos;
            } break;
            case CellType_Generator: {
                cellTO.cellTypeData.generator.additive = cell.cellTypeData.generator.additive;
                cellTO.cellTypeData.generator.minValue = cell.cellTypeData.generator.minValue;
                cellTO.cellTypeData.generator.maxValue = cell.cellTypeData.generator.maxValue;
                cellTO.cellTypeData.generator.timeOffset = cell.cellTypeData.generator.timeOffset;
                cellTO.cellTypeData.generator.mode = cell.cellTypeData.generator.mode;
                if (cell.cellTypeData.generator.mode == GeneratorMode_SquareSignal) {
                    cellTO.cellTypeData.generator.modeData.squareSignal.period = cell.cellTypeData.generator.modeData.squareSignal.period;
                } else if (cell.cellTypeData.generator.mode == GeneratorMode_SawtoothSignal) {
                    cellTO.cellTypeData.generator.modeData.sawtoothSignal.period = cell.cellTypeData.generator.modeData.sawtoothSignal.period;
                }
                cellTO.cellTypeData.generator.numPulses = cell.cellTypeData.generator.numPulses;
            } break;
            case CellType_Attacker: {
                cellTO.cellTypeData.attacker.mode = cell.cellTypeData.attacker.mode;
                if (cell.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                    cellTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors = cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors;
                }
            } break;
            case CellType_Injector: {
                cellTO.cellTypeData.injector.geneIndex = cell.cellTypeData.injector.geneIndex;
            } break;
            case CellType_Muscle: {
                cellTO.cellTypeData.muscle.mode = cell.cellTypeData.muscle.mode;
                if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
                    cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = cell.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                    cellTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = cell.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                    cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle = cell.cellTypeData.muscle.modeData.autoBending.initialAngle;
                    cellTO.cellTypeData.muscle.modeData.autoBending.forward = cell.cellTypeData.muscle.modeData.autoBending.forward;
                } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                    cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = cell.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                    cellTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio =
                        cell.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                    cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle = cell.cellTypeData.muscle.modeData.manualBending.initialAngle;
                    cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta = cell.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
                } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                    cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = cell.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                    cellTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                        cell.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                    cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle = cell.cellTypeData.muscle.modeData.angleBending.initialAngle;
                } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                    cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = cell.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                    cellTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = cell.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                    cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance = cell.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
                    cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = cell.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                    cellTO.cellTypeData.muscle.modeData.autoCrawling.forward = cell.cellTypeData.muscle.modeData.autoCrawling.forward;
                } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                        cell.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                        cell.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance = cell.cellTypeData.muscle.modeData.manualCrawling.initialDistance;
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = cell.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = cell.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
                } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
                }
                cellTO.cellTypeData.muscle.lastMovementX = cell.cellTypeData.muscle.lastMovementX;
                cellTO.cellTypeData.muscle.lastMovementY = cell.cellTypeData.muscle.lastMovementY;
            } break;
            case CellType_Defender: {
                cellTO.cellTypeData.defender.mode = cell.cellTypeData.defender.mode;
            } break;
            case CellType_Reconnector: {
                cellTO.cellTypeData.reconnector.mode = cell.cellTypeData.reconnector.mode;
                if (cell.cellTypeData.reconnector.mode == ReconnectorMode_Solid) {
                } else if (cell.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                    cellTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors =
                        cell.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors;
                } else if (cell.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                    cellTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells =
                        cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                    cellTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells =
                        cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                    cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors =
                        cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors;
                    cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage =
                        cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
                }
            } break;
            case CellType_Detonator: {
                cellTO.cellTypeData.detonator.state = cell.cellTypeData.detonator.state;
                cellTO.cellTypeData.detonator.countdown = cell.cellTypeData.detonator.countdown;
            } break;
            case CellType_Digestor: {
                cellTO.cellTypeData.digestor.rawEnergyConductivity = cell.cellTypeData.digestor.rawEnergyConductivity;
            } break;
            case CellType_Memory: {
                cellTO.cellTypeData.memory.mode = cell.cellTypeData.memory.mode;
                cellTO.cellTypeData.memory.numSignalEntries = cell.cellTypeData.memory.numSignalEntries;
                cellTO.cellTypeData.memory.channelBitMask = cell.cellTypeData.memory.channelBitMask;
                if (cell.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                    cellTO.cellTypeData.memory.modeData.signalDelay.delay = cell.cellTypeData.memory.modeData.signalDelay.delay;
                    cellTO.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized =
                        cell.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized;
                    cellTO.cellTypeData.memory.modeData.signalDelay.ringBufferIndex = cell.cellTypeData.memory.modeData.signalDelay.ringBufferIndex;
                } else if (cell.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                    cellTO.cellTypeData.memory.modeData.signalRecorder.readOnly = cell.cellTypeData.memory.modeData.signalRecorder.readOnly;
                    cellTO.cellTypeData.memory.modeData.signalRecorder.state = cell.cellTypeData.memory.modeData.signalRecorder.state;
                    cellTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries =
                        cell.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
                    cellTO.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries =
                        cell.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries;
                } else if (cell.cellTypeData.memory.mode == MemoryMode_SignalStorage) {
                    cellTO.cellTypeData.memory.modeData.signalStorage.readOnly = cell.cellTypeData.memory.modeData.signalStorage.readOnly;
                } else if (cell.cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
                    cellTO.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight = cell.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
                }
                int targetSize;  // not used
                copyDataToHeap<int>(
                    sizeof(SignalEntry) * cell.cellTypeData.memory.numSignalEntries,
                    reinterpret_cast<uint8_t*>(cell.cellTypeData.memory.signalEntries),
                    targetSize,
                    cellTO.cellTypeData.memory.signalEntriesDataIndex,
                    to);
            } break;
            case CellType_Communicator: {
                cellTO.cellTypeData.communicator.mode = cell.cellTypeData.communicator.mode;
                if (cell.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                    cellTO.cellTypeData.communicator.modeData.sender.range = cell.cellTypeData.communicator.modeData.sender.range;
                    cellTO.cellTypeData.communicator.modeData.sender.maxTimesSent = cell.cellTypeData.communicator.modeData.sender.maxTimesSent;
                } else if (cell.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                    cellTO.cellTypeData.communicator.modeData.receiver.restrictToColors = cell.cellTypeData.communicator.modeData.receiver.restrictToColors;
                    cellTO.cellTypeData.communicator.modeData.receiver.restrictToLineage = cell.cellTypeData.communicator.modeData.receiver.restrictToLineage;
                }
            } break;
            }

            // Handle optional constructor field
            cellTO.constructorAvailable = cell.constructorAvailable;
            if (cell.constructorAvailable) {
                cellTO.constructor.autoTriggerInterval = cell.constructor.autoTriggerInterval;
                cellTO.constructor.constructionActivationTime = cell.constructor.constructionActivationTime;
                cellTO.constructor.constructionAngle = cell.constructor.constructionAngle;
                cellTO.constructor.provideEnergy = cell.constructor.provideEnergy;
                cellTO.constructor.reservedEnergy = cell.constructor.reservedEnergy;
                cellTO.constructor.separation = cell.constructor.separation;
                cellTO.constructor.numBranches = cell.constructor.numBranches;
                cellTO.constructor.numConcatenations = cell.constructor.numConcatenations;
                cellTO.constructor.geneIndex = cell.constructor.geneIndex;
                cellTO.constructor.lastConstructedCellId = cell.constructor.lastConstructedCellId;
                cellTO.constructor.currentOffspring = cell.constructor.currentOffspring;
            }
        }
    }

    __device__ void createEnergyTO(Energy* particle, TOs& to)
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
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto pos = object->pos;
        data.objectMap.correctPosition(pos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            object->typeData.cell.creature->creatureIndex = VALUE_NOT_SET_UINT64;
            object->typeData.cell.creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaPrepareSelectedCreaturesForConversionToTO(bool includeClusters, SimulationData data)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            continue;
        }
        object->typeData.cell.creature->creatureIndex = VALUE_NOT_SET_UINT64;
        object->typeData.cell.creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
    }
}

__global__ void cudaPrepareCreaturesAndGenomesForConversionToTO(InspectedEntityIds ids, SimulationData data)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        object->typeData.cell.creature->creatureIndex = VALUE_NOT_SET_UINT64;
        object->typeData.cell.creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
    }
}

__global__ void cudaGetSelectedObjectDataWithoutConnections(SimulationData data, bool includeClusters, TOs to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    auto const heap = data.entities.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            object->tempValue1.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }
        createObjectTO(object, to, heap);
    }
}

__global__ void cudaGetSelectedEnergyData(SimulationData data, TOs access)
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

__global__ void cudaGetInspectedObjectDataWithoutConnections(InspectedEntityIds ids, SimulationData data, TOs to)
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
            object->tempValue1.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }

        createObjectTO(object, to, heapStart);
    }
}

__global__ void cudaGetInspectedEnergyData(InspectedEntityIds ids, SimulationData data, TOs access)
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

__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to)
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
            objectTO.typeData.cell.cellType = object->typeData.cell.cellType;
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

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to)
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
        if (object->type != ObjectType_Cell) {
            continue;
        }
        createGenomeTO(object->typeData.cell.creature->genome, to);
    }
}

__global__ void cudaGetSelectedGenomeData(SimulationData data, bool includeClusters, TOs to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            continue;
        }
        if (object->type != ObjectType_Cell) {
            continue;
        }

        createGenomeTO(object->typeData.cell.creature->genome, to);
    }
}

__global__ void cudaGetGenomeData(InspectedEntityIds ids, SimulationData data, TOs to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
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
        createGenomeTO(object->typeData.cell.creature->genome, to);
    }
}

__global__ void cudaGetCreatureData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to)
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
        if (object->type != ObjectType_Cell) {
            continue;
        }
        createCreatureTO(object, to);
    }
}

__global__ void cudaGetSelectedCreatureData(SimulationData data, bool includeClusters, TOs to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if ((includeClusters && object->selected == 0) || (!includeClusters && object->selected != 1)) {
            continue;
        }
        if (object->type != ObjectType_Cell) {
            continue;
        }

        createCreatureTO(object, to);
    }
}

__global__ void cudaGetCreatureData(InspectedEntityIds ids, SimulationData data, TOs to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
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

// tags cell with objectTO index and tags objectTO connections with cell index
__global__ void cudaGetObjectDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to)
{
    auto const& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    auto const heap = data.entities.heap.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        auto pos = object->pos;
        data.objectMap.correctPosition(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            object->tempValue1.as_uint64 = VALUE_NOT_SET_UINT64;
            continue;
        }

        createObjectTO(object, to, heap);
    }
}

__global__ void cudaResolveConnections(SimulationData data, TOs to)
{
    auto const partition = calcSystemThreadPartition(*to.numObjects);

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& objectTO = to.objects[index];

        for (int i = 0; i < objectTO.numConnections; ++i) {
            auto const objectIndex = objectTO.connections[i].objectIndex;
            objectTO.connections[i].objectIndex = data.entities.heap.atType<Object>(objectIndex).tempValue1.as_uint64;
        }
    }
}

__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs access)
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

__global__ void cudaGetArraysBasedOnTO(SimulationData data, TOs to, Object** cellArray)
{
    *cellArray = data.entities.heap.getTypedSubArray<Object>(*to.numObjects);
}

__global__ void cudaSetGenomeDataFromTO(SimulationData data, TOs to)
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

__global__ void cudaSetCreatureDataFromTO(SimulationData data, TOs to)
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

__global__ void cudaSetCellAndParticleDataFromTO(SimulationData data, TOs to, Object** cellArray, bool selectNewData)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto energyPartition = calcSystemThreadPartition(*to.numEnergyParticles);
    for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
        auto particle = factory.createParticleFromTO(to.energyParticles[index]);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto objectPartition = calcSystemThreadPartition(*to.numObjects);
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto object = factory.createObjectFromTO(to, index, *cellArray);
        if (selectNewData) {
            object->selected = 1;
        }
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, TOs to)
{
    Ids maxIds;
    {
        auto const partition = calcSystemThreadPartition(*to.numObjects);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& object = to.objects[index];
            maxIds.entityId = max(maxIds.entityId, object.id);
        }
    }
    {
        auto const partition = calcSystemThreadPartition(*to.numEnergyParticles);

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = to.energyParticles[index];
            maxIds.entityId = max(maxIds.entityId, particle.id);
        }
    }
    {
        auto const partition = calcSystemThreadPartition(*to.numCreatures);

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& creature = to.creatures[index];
            maxIds.entityId = max(maxIds.entityId, creature.id);
        }
    }
    {
        auto const partition = calcSystemThreadPartition(*to.numGenomes);

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& genome = to.genomes[index];
            maxIds.entityId = max(maxIds.entityId, genome.id);
            maxIds.lineageId = max(maxIds.lineageId, genome.lineageId);
        }
    }
    numberGen.adaptMaxIds(maxIds);
}

__global__ void cudaClearDataTO(TOs to)
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
        if (object->type == ObjectType_Cell) {
            object->typeData.cell.creature->creatureIndex = VALUE_NOT_SET_UINT64;
            object->typeData.cell.creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaEstimateCapacityNeededForTO_step2(SimulationData data, ArraySizesForTOs* arraySizes)
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
        if (object->type == ObjectType_Cell) {
            heapBytes += sizeof(NeuralNet) + GpuMemoryAlignmentBytes;
            if (object->typeData.cell.cellType == CellType_Memory) {
                heapBytes += sizeof(SignalEntry) * object->typeData.cell.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
            }
            if (object->typeData.cell.creature) {
                auto const& creature = object->typeData.cell.creature;
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
                                if (node.cellType == CellType_Memory) {
                                    heapBytes += sizeof(SignalEntryGenome) * node.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
                                }
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

__global__ void cudaEstimateCapacityNeededForGpu(TOs to, ArraySizesForGpuEntities* arraySizes)
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
            if (objectTO.type == ObjectType_Cell) {
                heapBytes += sizeof(NeuralNet) + GpuMemoryAlignmentBytes;
                if (objectTO.typeData.cell.cellType == CellType_Memory) {
                    heapBytes += sizeof(SignalEntry) * objectTO.typeData.cell.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
                }
            }
        }
        alienAtomicAdd64(&arraySizes->heap, heapBytes);
    }

    {
        auto partition = calcSystemThreadPartition(*to.numNodes);
        uint64_t heapBytes = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& nodeTO = to.nodes[index];
            if (nodeTO.cellType == CellType_Memory) {
                heapBytes += sizeof(SignalEntryGenome) * nodeTO.cellTypeData.memory.numSignalEntries + GpuMemoryAlignmentBytes;
            }
        }
        alienAtomicAdd64(&arraySizes->heap, heapBytes);
    }
}
