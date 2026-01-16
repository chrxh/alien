#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "Map.cuh"
#include "Entity.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"
#include "TO.cuh"

class EntityFactory
{
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Energy* createParticleFromTO(EnergyTO const& particleTO);
    __inline__ __device__ Creature* createCreatureFromTO(TO const& to, int creatureIndex);
    __inline__ __device__ Genome* createGenomeFromTO(TO const& to, int genomeIndex);
    __inline__ __device__ Object* createObjectFromTO(TO const& to, int objectIndex, Object* objectArray);
    __inline__ __device__ void changeObjectFromTO(TO const& to, ObjectTO const& objectTO, Object* object);
    __inline__ __device__ void changeEnergyFromTO(EnergyTO const& particleTO, Energy* particle);

    __inline__ __device__ Energy* createEnergy(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Object* createFreeCell(float energy, float2 const& pos, float2 const& vel);

    __inline__ __device__ Creature* cloneCreature(Creature* creature);

    __inline__ __device__ Object* createCellFromNode(
        uint64_t& objectIndex,
        Creature* creature,
        int geneIndex,
        int nodeIndex,
        int parentNodeIndex,
        float2 pos,
        float2 vel,
        float energy);
    __inline__ __device__ Creature* createEmptyCreature();
    __inline__ __device__ Gene* createEmptyGenes(int numGenes);
    __inline__ __device__ Node* createEmptyNodes(int numNodes);

private:
    template <typename T>
    __inline__ __device__ void copyDataToHeap(T sourceSize, uint64_t sourceIndex, uint8_t* heap, T& targetSize, uint8_t*& target);
    __inline__ __device__ void copyDataToHeap(uint64_t size, uint64_t sourceIndex, uint8_t* source, uint8_t*& target);

    BaseMap _map;
    SimulationData* _data;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void EntityFactory::init(SimulationData* data)
{
    _data = data;
    _map.init(data->worldSize);
}

__inline__ __device__ Energy* EntityFactory::createParticleFromTO(EnergyTO const& particleTO)
{
    Energy** particlePointer = _data->entities.energies.getNewElement();
    Energy* particle = _data->entities.heap.getTypedSubArray<Energy>(1);
    *particlePointer = particle;

    particle->id = particleTO.id;
    particle->pos = particleTO.pos;
    _map.correctPosition(particle->pos);
    particle->vel = particleTO.vel;
    particle->energy = particleTO.energy;
    particle->locked = 0;
    particle->selected = 0;
    particle->color = particleTO.color;
    particle->lastAbsorbedObject = nullptr;
    return particle;
}

__inline__ __device__ Genome* EntityFactory::createGenomeFromTO(TO const& to, int genomeIndex)
{
    auto& genomeTO = to.genomes[genomeIndex];
    auto genome = _data->entities.heap.getTypedSubArray<Genome>(1);
    genomeTO.genomeIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(genome) - _data->entities.heap.getArray());
    genome->id = genomeTO.id;
    genome->frontAngle = genomeTO.frontAngle;
    genome->numGenes = genomeTO.numGenes;
    for (int i = 0; i < sizeof(genomeTO.name); ++i) {
        genome->name[i] = genomeTO.name[i];
    }

    auto const& geneTOs = to.genes + genomeTO.geneArrayIndex;
    auto genes = _data->entities.heap.getTypedSubArray<Gene>(genomeTO.numGenes);
    genome->genes = genes;
    for (int i = 0, j = genomeTO.numGenes; i < j; ++i) {
        auto const& geneTO = geneTOs[i];
        auto& gene = genes[i];
        gene.shape = geneTO.shape;
        gene.separation = geneTO.separation;
        gene.numBranches = geneTO.numBranches;
        gene.angleAlignment = geneTO.angleAlignment;
        gene.stiffness = geneTO.stiffness;
        gene.connectionDistance = geneTO.connectionDistance;
        gene.numConcatenations = geneTO.numConcatenations;
        gene.numNodes = geneTO.numNodes;
        for (int i = 0; i < sizeof(geneTO.name); ++i) {
            gene.name[i] = geneTO.name[i];
        }

        auto const& nodeTOs = to.nodes + geneTO.nodeArrayIndex;
        auto nodes = _data->entities.heap.getTypedSubArray<Node>(geneTO.numNodes);
        gene.nodes = nodes;
        for (int i = 0, j = geneTO.numNodes; i < j; ++i) {
            auto const& nodeTO = nodeTOs[i];
            auto& node = nodes[i];
            node.referenceAngle = nodeTO.referenceAngle;
            node.color = nodeTO.color;
            node.numAdditionalConnections = nodeTO.numAdditionalConnections;
            for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
                node.neuralNetwork.weights[i] = nodeTO.neuralNetwork.weights[i];
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                node.neuralNetwork.biases[i] = nodeTO.neuralNetwork.biases[i];
                node.neuralNetwork.activationFunctions[i] = nodeTO.neuralNetwork.activationFunctions[i];
            }
            node.signalRestriction.mode = nodeTO.signalRestriction.mode;
            node.signalRestriction.baseAngle = nodeTO.signalRestriction.baseAngle;
            node.signalRestriction.openingAngle = nodeTO.signalRestriction.openingAngle;

            node.cellType = nodeTO.cellType;

            switch (nodeTO.cellType) {
            case CellTypeGenome_Base:
                break;
            case CellTypeGenome_Depot:
                node.cellTypeData.depot.storageLimit = nodeTO.cellTypeData.depot.storageLimit;
                node.cellTypeData.depot.initialStoredUsableEnergy = nodeTO.cellTypeData.depot.initialStoredUsableEnergy;
                break;
            case CellTypeGenome_Constructor:
                node.cellTypeData.constructor.autoTriggerInterval = nodeTO.cellTypeData.constructor.autoTriggerInterval;
                node.cellTypeData.constructor.geneIndex = nodeTO.cellTypeData.constructor.geneIndex;
                node.cellTypeData.constructor.constructionActivationTime = nodeTO.cellTypeData.constructor.constructionActivationTime;
                node.cellTypeData.constructor.constructionAngle = nodeTO.cellTypeData.constructor.constructionAngle;
                node.cellTypeData.constructor.provideEnergy = nodeTO.cellTypeData.constructor.provideEnergy;
                break;
            case CellTypeGenome_Sensor:
                node.cellTypeData.sensor.autoTriggerInterval = nodeTO.cellTypeData.sensor.autoTriggerInterval;
                node.cellTypeData.sensor.minRange = nodeTO.cellTypeData.sensor.minRange;
                node.cellTypeData.sensor.maxRange = nodeTO.cellTypeData.sensor.maxRange;
                node.cellTypeData.sensor.mode = nodeTO.cellTypeData.sensor.mode;
                if (nodeTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                    node.cellTypeData.sensor.modeData.detectEnergy.minDensity = nodeTO.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                    node.cellTypeData.sensor.modeData.detectFreeCell.minDensity = nodeTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                    node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor = nodeTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                    node.cellTypeData.sensor.modeData.detectCreature.minNumCells = nodeTO.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                    node.cellTypeData.sensor.modeData.detectCreature.maxNumCells = nodeTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                    node.cellTypeData.sensor.modeData.detectCreature.restrictToColor = nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor;
                    node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage = nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                }
                break;
            case CellTypeGenome_Generator:
                node.cellTypeData.generator.autoTriggerInterval = nodeTO.cellTypeData.generator.autoTriggerInterval;
                node.cellTypeData.generator.pulseType = nodeTO.cellTypeData.generator.pulseType;
                node.cellTypeData.generator.alternationInterval = nodeTO.cellTypeData.generator.alternationInterval;
                break;
            case CellTypeGenome_Attacker:
                node.cellTypeData.attacker.mode = nodeTO.cellTypeData.attacker.mode;
                if (nodeTO.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                    node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor = nodeTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
                } else if (nodeTO.cellTypeData.attacker.mode == AttackerMode_Creature) {
                    node.cellTypeData.attacker.modeData.attackCreature.minNumCells = nodeTO.cellTypeData.attacker.modeData.attackCreature.minNumCells;
                    node.cellTypeData.attacker.modeData.attackCreature.maxNumCells = nodeTO.cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                    node.cellTypeData.attacker.modeData.attackCreature.restrictToColor = nodeTO.cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                    node.cellTypeData.attacker.modeData.attackCreature.restrictToLineage = nodeTO.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
                }
                break;
            case CellTypeGenome_Injector:
                node.cellTypeData.injector.geneIndex = nodeTO.cellTypeData.injector.geneIndex;
                break;
            case CellTypeGenome_Muscle:
                node.cellTypeData.muscle.mode = nodeTO.cellTypeData.muscle.mode;
                switch (nodeTO.cellTypeData.muscle.mode) {
                case MuscleMode_AutoBending:
                    node.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = nodeTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                    node.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = nodeTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                    break;
                case MuscleMode_ManualBending:
                    node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = nodeTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                    node.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio =
                        nodeTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                    break;
                case MuscleMode_AngleBending:
                    node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = nodeTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                    node.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                        nodeTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                    break;
                case MuscleMode_AutoCrawling:
                    node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = nodeTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                    node.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = nodeTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                    break;
                case MuscleMode_ManualCrawling:
                    node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                        nodeTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                    node.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                        nodeTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                    break;
                case MuscleMode_DirectMovement:
                    break;
                }
                break;
            case CellTypeGenome_Defender:
                node.cellTypeData.defender.mode = nodeTO.cellTypeData.defender.mode;
                break;
            case CellTypeGenome_Reconnector:
                node.cellTypeData.reconnector.mode = nodeTO.cellTypeData.reconnector.mode;
                if (nodeTO.cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
                } else if (nodeTO.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                    node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor = nodeTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor;
                } else if (nodeTO.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                    node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells = nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                    node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells = nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                    node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor = nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor;
                    node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage = nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
                }
                break;
            case CellTypeGenome_Detonator:
                node.cellTypeData.detonator.countdown = nodeTO.cellTypeData.detonator.countdown;
                break;
            case CellTypeGenome_Digestor:
                node.cellTypeData.digestor.rawEnergyConductivity = nodeTO.cellTypeData.digestor.rawEnergyConductivity;
                break;
            case CellTypeGenome_Memory: {
                node.cellTypeData.memory.mode = nodeTO.cellTypeData.memory.mode;
                node.cellTypeData.memory.numSignalEntries = nodeTO.cellTypeData.memory.numSignalEntries;
                node.cellTypeData.memory.channelBitMask = nodeTO.cellTypeData.memory.channelBitMask;
                if (nodeTO.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                    node.cellTypeData.memory.modeData.signalDelay.delay = nodeTO.cellTypeData.memory.modeData.signalDelay.delay;
                } else if (nodeTO.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                    node.cellTypeData.memory.modeData.signalRecorder.readOnly = nodeTO.cellTypeData.memory.modeData.signalRecorder.readOnly;
                    node.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries = nodeTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
                } else if (nodeTO.cellTypeData.memory.mode == MemoryMode_SignalStorage) {
                    node.cellTypeData.memory.modeData.signalStorage.readOnly = nodeTO.cellTypeData.memory.modeData.signalStorage.readOnly;
                } else if (nodeTO.cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
                    node.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight = nodeTO.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
                }
                auto const& numSignalEntries = nodeTO.cellTypeData.memory.numSignalEntries;
                auto signalEntries = _data->entities.heap.getTypedSubArray<SignalEntryGenome>(numSignalEntries);
                node.cellTypeData.memory.signalEntries = signalEntries;
                auto const& entriesTO = reinterpret_cast<SignalEntryGenomeTO*>(to.heap + nodeTO.cellTypeData.memory.signalEntriesDataIndex);
                for (int k = 0; k < numSignalEntries; ++k) {
                    for (int l = 0; l < MAX_CHANNELS; ++l) {
                        signalEntries[k].channels[l] = entriesTO[k].channels[l];
                    }
                }
            } break;
            case CellTypeGenome_Communicator:
                node.cellTypeData.communicator.mode = nodeTO.cellTypeData.communicator.mode;
                if (nodeTO.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                    node.cellTypeData.communicator.modeData.sender.range = nodeTO.cellTypeData.communicator.modeData.sender.range;
                    node.cellTypeData.communicator.modeData.sender.maxTimesSent = nodeTO.cellTypeData.communicator.modeData.sender.maxTimesSent;
                } else if (nodeTO.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                    node.cellTypeData.communicator.modeData.receiver.restrictToColor = nodeTO.cellTypeData.communicator.modeData.receiver.restrictToColor;
                    node.cellTypeData.communicator.modeData.receiver.restrictToLineage = nodeTO.cellTypeData.communicator.modeData.receiver.restrictToLineage;
                }
                break;
            }
        }
    }
    return genome;
}

__inline__ __device__ Creature* EntityFactory::createCreatureFromTO(TO const& to, int creatureIndex)
{
    auto& creatureTO = to.creatures[creatureIndex];
    auto creature = _data->entities.heap.getTypedSubArray<Creature>(1);
    creatureTO.creatureIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(creature) - _data->entities.heap.getArray());

    creature->id = creatureTO.id;
    creature->ancestorId = creatureTO.ancestorId;
    creature->generation = creatureTO.generation;
    creature->lineageId = creatureTO.lineageId;
    creature->numObjects = creatureTO.numObjects;
    creature->frontAngleId = creatureTO.frontAngleId;

    auto const& genomeTO = to.genomes[creatureTO.genomeArrayIndex];
    creature->genome = &_data->entities.heap.atType<Genome>(genomeTO.genomeIndexOnGpu);

    return creature;
}

__inline__ __device__ Object* EntityFactory::createObjectFromTO(TO const& to, int objectIndex, Object* objectArray)
{
    auto objectTO = to.objects[objectIndex];
    Object** objectPointer = _data->entities.objects.getNewElement();
    Object* object = objectArray + objectIndex;
    *objectPointer = object;

    changeObjectFromTO(to, objectTO, object);
    object->id = objectTO.id;
    object->locked = 0;
    object->detached = 0;
    object->selected = 0;
    object->scheduledOperationIndex = -1;
    object->numConnections = objectTO.numConnections;
    object->event = CellEvent_No;
    object->density = 1.0f;
    for (int i = 0; i < object->numConnections; ++i) {
        auto& connectingCell = object->connections[i];
        connectingCell.object = objectArray + objectTO.connections[i].objectIndex;
        connectingCell.distance = objectTO.connections[i].distance;
        connectingCell.angleFromPrevious = objectTO.connections[i].angleFromPrevious;
    }
    if (objectTO.belongToCreature) {
        auto const& genomeTO = to.creatures[objectTO.creatureIndex];
        object->creature = &_data->entities.heap.atType<Creature>(genomeTO.creatureIndexOnGpu);
    } else {
        object->creature = nullptr;
    }
    return object;
}

__inline__ __device__ void EntityFactory::changeObjectFromTO(TO const& to, ObjectTO const& objectTO, Object* object)
{
    object->id = objectTO.id;
    object->pos = objectTO.pos;
    _map.correctPosition(object->pos);
    object->vel = objectTO.vel;
    object->cellState = objectTO.cellState;
    object->usableEnergy = objectTO.usableEnergy;
    object->rawEnergy = objectTO.rawEnergy;
    object->stiffness = objectTO.stiffness;
    object->cellType = objectTO.cellType;
    object->fixed = objectTO.fixed;
    object->sticky = objectTO.sticky;
    object->age = objectTO.age;
    object->color = objectTO.color;
    object->frontAngle = objectTO.frontAngle;
    object->activationTime = objectTO.activationTime;
    object->cellTriggered = objectTO.cellTriggered;
    object->nodeIndex = objectTO.nodeIndex;
    object->parentNodeIndex = objectTO.parentNodeIndex;
    object->geneIndex = objectTO.geneIndex;
    object->frontAngleId = objectTO.frontAngleId;
    object->headCell = objectTO.headCell;

    object->signalRestriction.mode = objectTO.signalRestriction.mode;
    object->signalRestriction.baseAngle = objectTO.signalRestriction.baseAngle;
    object->signalRestriction.openingAngle = objectTO.signalRestriction.openingAngle;

    object->signalState = objectTO.signalState;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->signal.channels[i] = objectTO.signal.channels[i];
    }
    object->signal.numTimesSent = objectTO.signal.numTimesSent;

    object->cellType = objectTO.cellType;

    if (objectTO.neuralNetworkDataIndex != VALUE_NOT_SET_UINT64) {
        copyDataToHeap(sizeof(NeuralNetworkTO), objectTO.neuralNetworkDataIndex, to.heap, reinterpret_cast<uint8_t*&>(object->neuralNetwork));
    } else {
        object->neuralNetwork = nullptr;
    }
    switch (objectTO.cellType) {
    case CellType_Base: {
    } break;
    case CellType_Depot: {
        object->cellTypeData.depot.storageLimit = objectTO.cellTypeData.depot.storageLimit;
        object->cellTypeData.depot.storedUsableEnergy = objectTO.cellTypeData.depot.storedUsableEnergy;
    } break;
    case CellType_Constructor: {
        object->cellTypeData.constructor.autoTriggerInterval = objectTO.cellTypeData.constructor.autoTriggerInterval;
        object->cellTypeData.constructor.constructionActivationTime = objectTO.cellTypeData.constructor.constructionActivationTime;
        object->cellTypeData.constructor.constructionAngle = objectTO.cellTypeData.constructor.constructionAngle;
        object->cellTypeData.constructor.provideEnergy = objectTO.cellTypeData.constructor.provideEnergy;
        object->cellTypeData.constructor.geneIndex = objectTO.cellTypeData.constructor.geneIndex;
        object->cellTypeData.constructor.lastConstructedCellId = objectTO.cellTypeData.constructor.lastConstructedCellId;
        object->cellTypeData.constructor.currentNodeIndex = objectTO.cellTypeData.constructor.currentNodeIndex;
        object->cellTypeData.constructor.currentConcatenation = objectTO.cellTypeData.constructor.currentConcatenation;
        object->cellTypeData.constructor.currentBranch = objectTO.cellTypeData.constructor.currentBranch;
        object->cellTypeData.constructor.isReady = true;
        object->cellTypeData.constructor.offspring = nullptr;
    } break;
    case CellType_Sensor: {
        object->cellTypeData.sensor.autoTriggerInterval = objectTO.cellTypeData.sensor.autoTriggerInterval;
        object->cellTypeData.sensor.minRange = objectTO.cellTypeData.sensor.minRange;
        object->cellTypeData.sensor.maxRange = objectTO.cellTypeData.sensor.maxRange;
        object->cellTypeData.sensor.mode = objectTO.cellTypeData.sensor.mode;
        if (objectTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
        } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
            object->cellTypeData.sensor.modeData.detectEnergy.minDensity = objectTO.cellTypeData.sensor.modeData.detectEnergy.minDensity;
        } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
        } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
            object->cellTypeData.sensor.modeData.detectFreeCell.minDensity = objectTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
            object->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor = objectTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
        } else if (objectTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
            object->cellTypeData.sensor.modeData.detectCreature.minNumCells = objectTO.cellTypeData.sensor.modeData.detectCreature.minNumCells;
            object->cellTypeData.sensor.modeData.detectCreature.maxNumCells = objectTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
            object->cellTypeData.sensor.modeData.detectCreature.restrictToColor = objectTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor;
            object->cellTypeData.sensor.modeData.detectCreature.restrictToLineage = objectTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
        }
        object->cellTypeData.sensor.lastMatchAvailable = objectTO.cellTypeData.sensor.lastMatchAvailable;
        object->cellTypeData.sensor.lastMatch.creatureId = objectTO.cellTypeData.sensor.lastMatch.creatureId;
        object->cellTypeData.sensor.lastMatch.pos = objectTO.cellTypeData.sensor.lastMatch.pos;
    } break;
    case CellType_Generator: {
        object->cellTypeData.generator.autoTriggerInterval = objectTO.cellTypeData.generator.autoTriggerInterval;
        object->cellTypeData.generator.pulseType = objectTO.cellTypeData.generator.pulseType;
        object->cellTypeData.generator.alternationInterval = objectTO.cellTypeData.generator.alternationInterval;
        object->cellTypeData.generator.numPulses = objectTO.cellTypeData.generator.numPulses;
    } break;
    case CellType_Attacker: {
        object->cellTypeData.attacker.mode = objectTO.cellTypeData.attacker.mode;
        if (objectTO.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
            object->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor = objectTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
        } else if (objectTO.cellTypeData.attacker.mode == AttackerMode_Creature) {
            object->cellTypeData.attacker.modeData.attackCreature.minNumCells = objectTO.cellTypeData.attacker.modeData.attackCreature.minNumCells;
            object->cellTypeData.attacker.modeData.attackCreature.maxNumCells = objectTO.cellTypeData.attacker.modeData.attackCreature.maxNumCells;
            object->cellTypeData.attacker.modeData.attackCreature.restrictToColor = objectTO.cellTypeData.attacker.modeData.attackCreature.restrictToColor;
            object->cellTypeData.attacker.modeData.attackCreature.restrictToLineage = objectTO.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
        }
    } break;
    case CellType_Injector: {
        object->cellTypeData.injector.geneIndex = objectTO.cellTypeData.injector.geneIndex;
    } break;
    case CellType_Muscle: {
        object->cellTypeData.muscle.mode = objectTO.cellTypeData.muscle.mode;
        if (objectTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
            object->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = objectTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            object->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = objectTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
            object->cellTypeData.muscle.modeData.autoBending.initialAngle = objectTO.cellTypeData.muscle.modeData.autoBending.initialAngle;
            object->cellTypeData.muscle.modeData.autoBending.forward = objectTO.cellTypeData.muscle.modeData.autoBending.forward;
            object->cellTypeData.muscle.modeData.autoBending.activation = objectTO.cellTypeData.muscle.modeData.autoBending.activation;
            object->cellTypeData.muscle.modeData.autoBending.activationCountdown = objectTO.cellTypeData.muscle.modeData.autoBending.activationCountdown;
            object->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = objectTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
        } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            object->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = objectTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            object->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio = objectTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
            object->cellTypeData.muscle.modeData.manualBending.initialAngle = objectTO.cellTypeData.muscle.modeData.manualBending.initialAngle;
            object->cellTypeData.muscle.modeData.manualBending.lastAngleDelta = objectTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
            object->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied = objectTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
        } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            object->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = objectTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            object->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                objectTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
            object->cellTypeData.muscle.modeData.angleBending.initialAngle = objectTO.cellTypeData.muscle.modeData.angleBending.initialAngle;
        } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
            object->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = objectTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            object->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = objectTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
            object->cellTypeData.muscle.modeData.autoCrawling.initialDistance = objectTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
            object->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = objectTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
            object->cellTypeData.muscle.modeData.autoCrawling.forward = objectTO.cellTypeData.muscle.modeData.autoCrawling.forward;
            object->cellTypeData.muscle.modeData.autoCrawling.activation = objectTO.cellTypeData.muscle.modeData.autoCrawling.activation;
            object->cellTypeData.muscle.modeData.autoCrawling.activationCountdown = objectTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
            object->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = objectTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
        } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
            object->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation = objectTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            object->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio = objectTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
            object->cellTypeData.muscle.modeData.manualCrawling.initialDistance = objectTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance;
            object->cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = objectTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
            object->cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = objectTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
            object->cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied = objectTO.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
        } else if (objectTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
        }
        object->cellTypeData.muscle.lastMovementX = objectTO.cellTypeData.muscle.lastMovementX;
        object->cellTypeData.muscle.lastMovementY = objectTO.cellTypeData.muscle.lastMovementY;
    } break;
    case CellType_Defender: {
        object->cellTypeData.defender.mode = objectTO.cellTypeData.defender.mode;
    } break;
    case CellType_Reconnector: {
        object->cellTypeData.reconnector.mode = objectTO.cellTypeData.reconnector.mode;
        if (objectTO.cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
        } else if (objectTO.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
            object->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor = objectTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor;
        } else if (objectTO.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
            object->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells = objectTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
            object->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells = objectTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
            object->cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor = objectTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor;
            object->cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage = objectTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
        }
    } break;
    case CellType_Detonator: {
        object->cellTypeData.detonator.state = objectTO.cellTypeData.detonator.state;
        object->cellTypeData.detonator.countdown = objectTO.cellTypeData.detonator.countdown;
    } break;
    case CellType_Digestor: {
        object->cellTypeData.digestor.rawEnergyConductivity = objectTO.cellTypeData.digestor.rawEnergyConductivity;
    } break;
    case CellType_Memory: {
        object->cellTypeData.memory.mode = objectTO.cellTypeData.memory.mode;
        object->cellTypeData.memory.numSignalEntries = objectTO.cellTypeData.memory.numSignalEntries;
        object->cellTypeData.memory.channelBitMask = objectTO.cellTypeData.memory.channelBitMask;
        if (objectTO.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
            object->cellTypeData.memory.modeData.signalDelay.delay = objectTO.cellTypeData.memory.modeData.signalDelay.delay;
            object->cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized = objectTO.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized;
            object->cellTypeData.memory.modeData.signalDelay.ringBufferIndex = objectTO.cellTypeData.memory.modeData.signalDelay.ringBufferIndex;
        } else if (objectTO.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
            object->cellTypeData.memory.modeData.signalRecorder.readOnly = objectTO.cellTypeData.memory.modeData.signalRecorder.readOnly;
            object->cellTypeData.memory.modeData.signalRecorder.state = objectTO.cellTypeData.memory.modeData.signalRecorder.state;
            object->cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries = objectTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
            object->cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries = objectTO.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries;
        } else if (objectTO.cellTypeData.memory.mode == MemoryMode_SignalStorage) {
            object->cellTypeData.memory.modeData.signalStorage.readOnly = objectTO.cellTypeData.memory.modeData.signalStorage.readOnly;
        } else if (objectTO.cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
            object->cellTypeData.memory.modeData.signalIntegrator.newSignalWeight = objectTO.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
        }
        copyDataToHeap(
            sizeof(SignalEntryTO) * objectTO.cellTypeData.memory.numSignalEntries,
            objectTO.cellTypeData.memory.signalEntriesDataIndex,
            to.heap,
            reinterpret_cast<uint8_t*&>(object->cellTypeData.memory.signalEntries));
    } break;
    case CellType_Communicator: {
        object->cellTypeData.communicator.mode = objectTO.cellTypeData.communicator.mode;
        if (objectTO.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
            object->cellTypeData.communicator.modeData.sender.range = objectTO.cellTypeData.communicator.modeData.sender.range;
            object->cellTypeData.communicator.modeData.sender.maxTimesSent = objectTO.cellTypeData.communicator.modeData.sender.maxTimesSent;
        } else if (objectTO.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
            object->cellTypeData.communicator.modeData.receiver.restrictToColor = objectTO.cellTypeData.communicator.modeData.receiver.restrictToColor;
            object->cellTypeData.communicator.modeData.receiver.restrictToLineage = objectTO.cellTypeData.communicator.modeData.receiver.restrictToLineage;
        }
    } break;
    }
}

__inline__ __device__ void EntityFactory::changeEnergyFromTO(EnergyTO const& particleTO, Energy* particle)
{
    particle->energy = particleTO.energy;
    particle->pos = particleTO.pos;
    particle->color = particleTO.color;
}

template <typename T>
__inline__ __device__ void EntityFactory::copyDataToHeap(T sourceSize, uint64_t sourceIndex, uint8_t* heap, T& targetSize, uint8_t*& target)
{
    targetSize = sourceSize;
    copyDataToHeap(sourceSize, sourceIndex, heap, target);
}

__inline__ __device__ void EntityFactory::copyDataToHeap(uint64_t size, uint64_t sourceIndex, uint8_t* source, uint8_t*& target)
{
    if (size > 0) {
        target = _data->entities.heap.getRawSubArray(size);
        for (int i = 0; i < size; ++i) {
            target[i] = source[sourceIndex + i];
        }
    }
}

__inline__ __device__ Energy* EntityFactory::createEnergy(float energy, float2 const& pos, float2 const& vel, int color)
{
    Energy** particlePointer = _data->entities.energies.getNewElement();
    Energy* particle = _data->entities.heap.getTypedSubArray<Energy>(1);
    *particlePointer = particle;
    particle->id = _data->primaryNumberGen.createId();
    particle->selected = 0;
    particle->locked = 0;
    particle->energy = energy;
    particle->pos = pos;
    particle->vel = vel;
    particle->color = color;
    particle->lastAbsorbedObject = nullptr;
    return particle;
}

__inline__ __device__ Object* EntityFactory::createFreeCell(float energy, float2 const& pos, float2 const& vel)
{
    auto object = _data->entities.heap.getTypedSubArray<Object>(1);
    auto objectPointers = _data->entities.objects.getNewElement();
    *objectPointers = object;

    object->id = _data->primaryNumberGen.createId();
    object->pos = pos;
    object->vel = vel;
    object->usableEnergy = energy;
    object->stiffness = _data->primaryNumberGen.random();
    object->numConnections = 0;
    object->cellState = CellState_Ready;
    object->locked = 0;
    object->selected = 0;
    object->detached = 0;
    object->scheduledOperationIndex = -1;
    object->color = 0;
    object->frontAngle = VALUE_NOT_SET_FLOAT;
    object->frontAngleId = 0;
    object->headCell = false;
    object->fixed = false;
    object->sticky = false;
    object->age = 0;
    object->activationTime = 0;
    object->signalRestriction.mode = SignalRestrictionMode_Inactive;
    object->signalState = 0;
    object->density = 1.0f;
    object->event = CellEvent_No;
    object->cellTriggered = CellTriggered_No;
    object->creature = nullptr;
    object->nodeIndex = 0;
    object->parentNodeIndex = 0;
    object->geneIndex = 0;
    object->cellType = CellType_Free;
    object->neuralNetwork = nullptr;

    return object;
}

__inline__ __device__ Creature* EntityFactory::cloneCreature(Creature* creature)
{
    auto newCreature = createEmptyCreature();
    auto newId = newCreature->id;
    *newCreature = *creature;
    newCreature->id = newId;
    newCreature->ancestorId = creature->id;
    newCreature->generation = creature->generation + 1;
    return newCreature;
}

//__inline__ __device__ Object* EntityFactory::createEmptyCell(uint64_t& objectIndex)
//{
//    auto cell = _data->entities.heap.getTypedSubArray<Object>(1);
//    auto objectPointer = _data->entities.objects.getNewElement(&objectIndex);
//    *objectPointer = object;
//
//    object->id = _data->primaryNumberGen.createObjectId();
//    object->stiffness = 1.0f;
//    object->selected = 0;
//    object->detached = 0;
//    object->scheduledOperationIndex = -1;
//    object->locked = 0;
//    object->color = 0;
//    object->fixed = false;
//    object->sticky = false;
//    object->age = 0;
//    object->vel = {0, 0};
//    object->activationTime = 0;
//    object->signalRestriction.active = false;
//    object->signalState = 0;
//    object->density = 1.0f;
//    object->event = CellEvent_No;
//    object->cellTriggered = CellTriggered_No;
//    return object;
//}

__inline__ __device__ Object* EntityFactory::createCellFromNode(
    uint64_t& objectIndex,
    Creature* creature,
    int geneIndex,
    int nodeIndex,
    int parentNodeIndex,
    float2 pos,
    float2 vel,
    float energy)
{
    auto const& gene = &creature->genome->genes[geneIndex];
    auto const& node = &gene->nodes[nodeIndex];

    auto object = _data->entities.heap.getTypedSubArray<Object>(1);
    auto objectPointer = _data->entities.objects.getNewElement(&objectIndex);
    *objectPointer = object;
    object->id = _data->primaryNumberGen.createId();
    object->pos = pos;
    object->vel = vel;
    object->usableEnergy = energy;
    object->rawEnergy = 0;
    object->stiffness = gene->stiffness;
    object->color = node->color;
    object->frontAngle = VALUE_NOT_SET_FLOAT;
    object->stiffness = 1.0f;
    object->fixed = false;
    object->sticky = false;
    object->age = 0;
    object->cellState = CellState_Constructing;
    object->creature = creature;
    object->nodeIndex = nodeIndex;
    object->parentNodeIndex = parentNodeIndex;
    object->geneIndex = geneIndex;
    object->numConnections = 0;
    object->frontAngleId = 0;
    object->headCell = false;

    object->neuralNetwork = _data->entities.heap.getTypedSubArray<NeuralNetwork>(1);
    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        object->neuralNetwork->weights[i] = node->neuralNetwork.weights[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->neuralNetwork->biases[i] = node->neuralNetwork.biases[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->neuralNetwork->activationFunctions[i] = node->neuralNetwork.activationFunctions[i];
    }
    object->signalRestriction.mode = node->signalRestriction.mode;
    object->signalRestriction.baseAngle = node->signalRestriction.baseAngle;
    object->signalRestriction.openingAngle = node->signalRestriction.openingAngle;
    object->signalState = 0;
    object->activationTime = 0;
    object->cellTriggered = CellTriggered_No;
    object->event = CellEvent_No;
    object->selected = 0;
    object->detached = 0;
    object->locked = 0;
    object->density = 1.0f;
    object->scheduledOperationIndex = -1;

    switch (node->cellType) {
    case CellTypeGenome_Base: {
        object->cellType = CellType_Base;
    } break;
    case CellTypeGenome_Depot: {
        object->cellType = CellType_Depot;
        object->cellTypeData.depot.storageLimit = node->cellTypeData.depot.storageLimit;
        object->cellTypeData.depot.storedUsableEnergy = node->cellTypeData.depot.initialStoredUsableEnergy;
    } break;
    case CellTypeGenome_Constructor: {
        object->cellType = CellType_Constructor;
        auto const& nodeConstructor = node->cellTypeData.constructor;
        auto& constructor = object->cellTypeData.constructor;
        constructor.autoTriggerInterval = nodeConstructor.autoTriggerInterval;
        constructor.constructionActivationTime = nodeConstructor.constructionActivationTime;
        constructor.constructionAngle = nodeConstructor.constructionAngle;
        constructor.provideEnergy = nodeConstructor.provideEnergy;
        constructor.geneIndex = nodeConstructor.geneIndex;
        constructor.lastConstructedCellId = VALUE_NOT_SET_UINT64;
        constructor.currentNodeIndex = 0;
        constructor.currentConcatenation = 0;
        constructor.currentBranch = 0;
        constructor.isReady = true;
        constructor.offspring = nullptr;
    } break;
    case CellTypeGenome_Sensor: {
        object->cellType = CellType_Sensor;
        auto const& nodeSensor = node->cellTypeData.sensor;
        auto& sensor = object->cellTypeData.sensor;
        sensor.autoTriggerInterval = nodeSensor.autoTriggerInterval;
        sensor.minRange = nodeSensor.minRange;
        sensor.maxRange = nodeSensor.maxRange;
        sensor.mode = nodeSensor.mode;
        if (nodeSensor.mode == SensorMode_Telemetry) {
        } else if (nodeSensor.mode == SensorMode_DetectEnergy) {
            sensor.modeData.detectEnergy.minDensity = nodeSensor.modeData.detectEnergy.minDensity;
        } else if (nodeSensor.mode == SensorMode_DetectStructure) {
        } else if (nodeSensor.mode == SensorMode_DetectFreeCell) {
            sensor.modeData.detectFreeCell.minDensity = nodeSensor.modeData.detectFreeCell.minDensity;
            sensor.modeData.detectFreeCell.restrictToColor = nodeSensor.modeData.detectFreeCell.restrictToColor;
        } else if (nodeSensor.mode == SensorMode_DetectCreature) {
            sensor.modeData.detectCreature.minNumCells = nodeSensor.modeData.detectCreature.minNumCells;
            sensor.modeData.detectCreature.maxNumCells = nodeSensor.modeData.detectCreature.maxNumCells;
            sensor.modeData.detectCreature.restrictToColor = nodeSensor.modeData.detectCreature.restrictToColor;
            sensor.modeData.detectCreature.restrictToLineage = nodeSensor.modeData.detectCreature.restrictToLineage;
        }
        sensor.lastMatchAvailable = false;
    } break;
    case CellTypeGenome_Generator: {
        object->cellType = CellType_Generator;
        auto const& nodeGenerator = node->cellTypeData.generator;
        auto& generator = object->cellTypeData.generator;
        generator.autoTriggerInterval = nodeGenerator.autoTriggerInterval;
        generator.pulseType = nodeGenerator.pulseType;
        generator.alternationInterval = nodeGenerator.alternationInterval;
        generator.numPulses = 0;
    } break;
    case CellTypeGenome_Attacker: {
        object->cellType = CellType_Attacker;
        auto const& nodeAttacker = node->cellTypeData.attacker;
        auto& attacker = object->cellTypeData.attacker;
        attacker.mode = nodeAttacker.mode;
        if (nodeAttacker.mode == AttackerMode_FreeCell) {
            attacker.modeData.attackFreeCell.restrictToColor = nodeAttacker.modeData.attackFreeCell.restrictToColor;
        } else if (nodeAttacker.mode == AttackerMode_Creature) {
            attacker.modeData.attackCreature.minNumCells = nodeAttacker.modeData.attackCreature.minNumCells;
            attacker.modeData.attackCreature.maxNumCells = nodeAttacker.modeData.attackCreature.maxNumCells;
            attacker.modeData.attackCreature.restrictToColor = nodeAttacker.modeData.attackCreature.restrictToColor;
            attacker.modeData.attackCreature.restrictToLineage = nodeAttacker.modeData.attackCreature.restrictToLineage;
        }
    } break;
    case CellTypeGenome_Injector: {
        object->cellType = CellType_Injector;
        auto const& nodeInjector = node->cellTypeData.injector;
        auto& injector = object->cellTypeData.injector;
        injector.geneIndex = nodeInjector.geneIndex;
    } break;
    case CellTypeGenome_Muscle: {
        object->cellType = CellType_Muscle;
        auto const& nodeMuscle = node->cellTypeData.muscle;
        auto& muscle = object->cellTypeData.muscle;
        muscle.lastMovementX = 0;
        muscle.lastMovementY = 0;
        muscle.mode = nodeMuscle.mode;
        switch (nodeMuscle.mode) {
        case MuscleMode_AutoBending: {
            muscle.modeData.autoBending.maxAngleDeviation = nodeMuscle.modeData.autoBending.maxAngleDeviation;
            muscle.modeData.autoBending.forwardBackwardRatio = nodeMuscle.modeData.autoBending.forwardBackwardRatio;
            muscle.modeData.autoBending.initialAngle = VALUE_NOT_SET_FLOAT;
            muscle.modeData.autoBending.forward = true;
            muscle.modeData.autoBending.activation = 0;
            muscle.modeData.autoBending.activationCountdown = 0;
            muscle.modeData.autoBending.impulseAlreadyApplied = false;
        } break;
        case MuscleMode_ManualBending: {
            muscle.modeData.manualBending.maxAngleDeviation = nodeMuscle.modeData.manualBending.maxAngleDeviation;
            muscle.modeData.manualBending.forwardBackwardRatio = nodeMuscle.modeData.manualBending.forwardBackwardRatio;
            muscle.modeData.manualBending.initialAngle = VALUE_NOT_SET_FLOAT;
            muscle.modeData.manualBending.lastAngleDelta = 0;
            muscle.modeData.manualBending.impulseAlreadyApplied = false;
        } break;
        case MuscleMode_AngleBending: {
            muscle.modeData.angleBending.maxAngleDeviation = nodeMuscle.modeData.angleBending.maxAngleDeviation;
            muscle.modeData.angleBending.attractionRepulsionRatio = nodeMuscle.modeData.angleBending.attractionRepulsionRatio;
            muscle.modeData.angleBending.initialAngle = VALUE_NOT_SET_FLOAT;
        } break;
        case MuscleMode_AutoCrawling: {
            muscle.modeData.autoCrawling.maxDistanceDeviation = nodeMuscle.modeData.autoCrawling.maxDistanceDeviation;
            muscle.modeData.autoCrawling.forwardBackwardRatio = nodeMuscle.modeData.autoCrawling.forwardBackwardRatio;
            muscle.modeData.autoCrawling.initialDistance = VALUE_NOT_SET_FLOAT;
            muscle.modeData.autoCrawling.lastActualDistance = 0;
            muscle.modeData.autoCrawling.forward = true;
            muscle.modeData.autoCrawling.activation = 0;
            muscle.modeData.autoCrawling.activationCountdown = 0;
            muscle.modeData.autoCrawling.impulseAlreadyApplied = false;
        } break;
        case MuscleMode_ManualCrawling: {
            muscle.modeData.manualCrawling.maxDistanceDeviation = nodeMuscle.modeData.manualCrawling.maxDistanceDeviation;
            muscle.modeData.manualCrawling.forwardBackwardRatio = nodeMuscle.modeData.manualCrawling.forwardBackwardRatio;
            muscle.modeData.manualCrawling.initialDistance = VALUE_NOT_SET_FLOAT;
            muscle.modeData.manualCrawling.lastActualDistance = 0;
            muscle.modeData.manualCrawling.lastDistanceDelta = 0;
            muscle.modeData.manualCrawling.impulseAlreadyApplied = false;
        } break;
        case MuscleMode_DirectMovement: {
        } break;
        }
    } break;
    case CellTypeGenome_Defender: {
        object->cellType = CellType_Defender;
        auto const& nodeDefender = node->cellTypeData.defender;
        auto& defender = object->cellTypeData.defender;
        defender.mode = nodeDefender.mode;
    } break;
    case CellTypeGenome_Reconnector: {
        object->cellType = CellType_Reconnector;
        auto const& nodeReconnector = node->cellTypeData.reconnector;
        auto& reconnector = object->cellTypeData.reconnector;
        reconnector.mode = nodeReconnector.mode;
        if (nodeReconnector.mode == ReconnectorMode_Structure) {
        } else if (nodeReconnector.mode == ReconnectorMode_FreeCell) {
            reconnector.modeData.reconnectFreeCell.restrictToColor = nodeReconnector.modeData.reconnectFreeCell.restrictToColor;
        } else if (nodeReconnector.mode == ReconnectorMode_Creature) {
            reconnector.modeData.reconnectCreature.minNumCells = nodeReconnector.modeData.reconnectCreature.minNumCells;
            reconnector.modeData.reconnectCreature.maxNumCells = nodeReconnector.modeData.reconnectCreature.maxNumCells;
            reconnector.modeData.reconnectCreature.restrictToColor = nodeReconnector.modeData.reconnectCreature.restrictToColor;
            reconnector.modeData.reconnectCreature.restrictToLineage = nodeReconnector.modeData.reconnectCreature.restrictToLineage;
        }
    } break;
    case CellTypeGenome_Detonator: {
        object->cellType = CellType_Detonator;
        auto const& nodeDetonator = node->cellTypeData.detonator;
        auto& detonator = object->cellTypeData.detonator;
        detonator.state = DetonatorState_Ready;
        detonator.countdown = nodeDetonator.countdown;
    } break;
    case CellTypeGenome_Digestor: {
        object->cellType = CellType_Digestor;
        auto const& nodeDigestor = node->cellTypeData.digestor;
        auto& digestor = object->cellTypeData.digestor;
        digestor.rawEnergyConductivity = nodeDigestor.rawEnergyConductivity;
    } break;
    case CellTypeGenome_Memory: {
        object->cellType = CellType_Memory;
        auto const& nodeMemory = node->cellTypeData.memory;
        auto& memory = object->cellTypeData.memory;
        memory.mode = nodeMemory.mode;
        memory.numSignalEntries = nodeMemory.numSignalEntries;
        memory.channelBitMask = nodeMemory.channelBitMask;
        if (nodeMemory.mode == MemoryMode_SignalDelay) {
            memory.modeData.signalDelay.delay = nodeMemory.modeData.signalDelay.delay;
            memory.modeData.signalDelay.numSignalEntriesInitialized = 0;
            memory.modeData.signalDelay.ringBufferIndex = 0;
        } else if (nodeMemory.mode == MemoryMode_SignalRecorder) {
            memory.modeData.signalRecorder.readOnly = nodeMemory.modeData.signalRecorder.readOnly;
            memory.modeData.signalRecorder.state = SignalRecorderState_Idle;
            memory.modeData.signalRecorder.numWrittenSignalEntries = nodeMemory.modeData.signalRecorder.numWrittenSignalEntries;
            memory.modeData.signalRecorder.numReadSignalEntries = 0;
        } else if (nodeMemory.mode == MemoryMode_SignalStorage) {
            memory.modeData.signalStorage.readOnly = nodeMemory.modeData.signalStorage.readOnly;
        } else if (nodeMemory.mode == MemoryMode_SignalIntegrator) {
            memory.modeData.signalIntegrator.newSignalWeight = nodeMemory.modeData.signalIntegrator.newSignalWeight;
        }
        memory.signalEntries = _data->entities.heap.getTypedSubArray<SignalEntry>(nodeMemory.numSignalEntries);
        for (int i = 0, j = nodeMemory.numSignalEntries; i < j; ++i) {
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                memory.signalEntries[i].channels[k] = nodeMemory.signalEntries[i].channels[k];
            }
        }
    } break;
    case CellTypeGenome_Communicator: {
        object->cellType = CellType_Communicator;
        auto const& nodeCommunicator = node->cellTypeData.communicator;
        auto& communicator = object->cellTypeData.communicator;
        communicator.mode = nodeCommunicator.mode;
        if (nodeCommunicator.mode == CommunicatorMode_Sender) {
            communicator.modeData.sender.range = nodeCommunicator.modeData.sender.range;
            communicator.modeData.sender.maxTimesSent = nodeCommunicator.modeData.sender.maxTimesSent;
        } else if (nodeCommunicator.mode == CommunicatorMode_Receiver) {
            communicator.modeData.receiver.restrictToColor = nodeCommunicator.modeData.receiver.restrictToColor;
            communicator.modeData.receiver.restrictToLineage = nodeCommunicator.modeData.receiver.restrictToLineage;
        }
    } break;
    }
    return object;
}

__inline__ __device__ Creature* EntityFactory::createEmptyCreature()
{
    auto creature = _data->entities.heap.getTypedSubArray<Creature>(1);
    creature->id = _data->primaryNumberGen.createId();
    return creature;
}

__inline__ __device__ Gene* EntityFactory::createEmptyGenes(int numGenes)
{
    auto genes = _data->entities.heap.getTypedSubArray<Gene>(numGenes);
    return genes;
}

__inline__ __device__ Node* EntityFactory::createEmptyNodes(int numNodes)
{
    auto nodes = _data->entities.heap.getTypedSubArray<Node>(numNodes);
    return nodes;
}
