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
    if (object->type == ObjectType_Cell) {
        object->typeData.cell.event = CellEvent_No;
        auto const& genomeTO = to.creatures[objectTO.typeData.cell.creatureIndex];
        object->typeData.cell.creature = &_data->entities.heap.atType<Creature>(genomeTO.creatureIndexOnGpu);
    } else if (object->type == ObjectType_FreeCell) {
        object->typeData.freeCell.event = CellEvent_No;
    }
    object->density = 1.0f;
    for (int i = 0; i < object->numConnections; ++i) {
        auto& connectedObject = object->connections[i];
        connectedObject.object = objectArray + objectTO.connections[i].objectIndex;
        connectedObject.distance = objectTO.connections[i].distance;
        connectedObject.angleFromPrevious = objectTO.connections[i].angleFromPrevious;
    }
    return object;
}

__inline__ __device__ void EntityFactory::changeObjectFromTO(TO const& to, ObjectTO const& objectTO, Object* object)
{
    object->id = objectTO.id;
    object->pos = objectTO.pos;
    _map.correctPosition(object->pos);
    object->vel = objectTO.vel;
    object->type = objectTO.type;

    if (objectTO.type == ObjectType_Structure) {
    } else if (objectTO.type == ObjectType_FreeCell) {
        object->typeData.freeCell.rawEnergy = objectTO.typeData.freeCell.rawEnergy; 
    } else if (objectTO.type == ObjectType_Cell) {
        object->typeData.cell.cellState = objectTO.typeData.cell.cellState;
        object->typeData.cell.usableEnergy = objectTO.typeData.cell.usableEnergy;
        object->typeData.cell.rawEnergy = objectTO.typeData.cell.rawEnergy;
        object->typeData.cell.cellType = objectTO.typeData.cell.cellType;
        object->typeData.cell.age = objectTO.typeData.cell.age;
        object->typeData.cell.frontAngle = objectTO.typeData.cell.frontAngle;
        object->typeData.cell.activationTime = objectTO.typeData.cell.activationTime;
        object->typeData.cell.cellTriggered = objectTO.typeData.cell.cellTriggered;
        object->typeData.cell.nodeIndex = objectTO.typeData.cell.nodeIndex;
        object->typeData.cell.parentNodeIndex = objectTO.typeData.cell.parentNodeIndex;
        object->typeData.cell.geneIndex = objectTO.typeData.cell.geneIndex;
        object->typeData.cell.frontAngleId = objectTO.typeData.cell.frontAngleId;
        object->typeData.cell.headCell = objectTO.typeData.cell.headCell;

        object->typeData.cell.signalRestriction.mode = objectTO.typeData.cell.signalRestriction.mode;
        object->typeData.cell.signalRestriction.baseAngle = objectTO.typeData.cell.signalRestriction.baseAngle;
        object->typeData.cell.signalRestriction.openingAngle = objectTO.typeData.cell.signalRestriction.openingAngle;

        object->typeData.cell.signalState = objectTO.typeData.cell.signalState;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            object->typeData.cell.signal.channels[i] = objectTO.typeData.cell.signal.channels[i];
        }
        object->typeData.cell.signal.numTimesSent = objectTO.typeData.cell.signal.numTimesSent;

        object->typeData.cell.cellType = objectTO.typeData.cell.cellType;

        copyDataToHeap(
            sizeof(NeuralNetworkTO),
            objectTO.typeData.cell.neuralNetworkDataIndex,
            to.heap,
            reinterpret_cast<uint8_t*&>(object->typeData.cell.neuralNetwork));

        switch (objectTO.typeData.cell.cellType) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            object->typeData.cell.cellTypeData.depot.storageLimit = objectTO.typeData.cell.cellTypeData.depot.storageLimit;
            object->typeData.cell.cellTypeData.depot.storedUsableEnergy = objectTO.typeData.cell.cellTypeData.depot.storedUsableEnergy;
        } break;
        case CellType_Constructor: {
            object->typeData.cell.cellTypeData.constructor.autoTriggerInterval = objectTO.typeData.cell.cellTypeData.constructor.autoTriggerInterval;
            object->typeData.cell.cellTypeData.constructor.constructionActivationTime =
                objectTO.typeData.cell.cellTypeData.constructor.constructionActivationTime;
            object->typeData.cell.cellTypeData.constructor.constructionAngle = objectTO.typeData.cell.cellTypeData.constructor.constructionAngle;
            object->typeData.cell.cellTypeData.constructor.provideEnergy = objectTO.typeData.cell.cellTypeData.constructor.provideEnergy;
            object->typeData.cell.cellTypeData.constructor.geneIndex = objectTO.typeData.cell.cellTypeData.constructor.geneIndex;
            object->typeData.cell.cellTypeData.constructor.lastConstructedCellId = objectTO.typeData.cell.cellTypeData.constructor.lastConstructedCellId;
            object->typeData.cell.cellTypeData.constructor.currentNodeIndex = objectTO.typeData.cell.cellTypeData.constructor.currentNodeIndex;
            object->typeData.cell.cellTypeData.constructor.currentConcatenation = objectTO.typeData.cell.cellTypeData.constructor.currentConcatenation;
            object->typeData.cell.cellTypeData.constructor.currentBranch = objectTO.typeData.cell.cellTypeData.constructor.currentBranch;
            object->typeData.cell.cellTypeData.constructor.isReady = true;
            object->typeData.cell.cellTypeData.constructor.offspring = nullptr;
        } break;
        case CellType_Sensor: {
            object->typeData.cell.cellTypeData.sensor.autoTriggerInterval = objectTO.typeData.cell.cellTypeData.sensor.autoTriggerInterval;
            object->typeData.cell.cellTypeData.sensor.minRange = objectTO.typeData.cell.cellTypeData.sensor.minRange;
            object->typeData.cell.cellTypeData.sensor.maxRange = objectTO.typeData.cell.cellTypeData.sensor.maxRange;
            object->typeData.cell.cellTypeData.sensor.mode = objectTO.typeData.cell.cellTypeData.sensor.mode;
            if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_Telemetry) {
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                object->typeData.cell.cellTypeData.sensor.modeData.detectEnergy.minDensity =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectEnergy.minDensity;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                object->typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.minDensity =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                object->typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                object->typeData.cell.cellTypeData.sensor.modeData.detectCreature.minNumCells =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                object->typeData.cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                object->typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToColor;
                object->typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToLineage =
                    objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            }
            object->typeData.cell.cellTypeData.sensor.lastMatchAvailable = objectTO.typeData.cell.cellTypeData.sensor.lastMatchAvailable;
            object->typeData.cell.cellTypeData.sensor.lastMatch.creatureId = objectTO.typeData.cell.cellTypeData.sensor.lastMatch.creatureId;
            object->typeData.cell.cellTypeData.sensor.lastMatch.pos = objectTO.typeData.cell.cellTypeData.sensor.lastMatch.pos;
        } break;
        case CellType_Generator: {
            object->typeData.cell.cellTypeData.generator.autoTriggerInterval = objectTO.typeData.cell.cellTypeData.generator.autoTriggerInterval;
            object->typeData.cell.cellTypeData.generator.pulseType = objectTO.typeData.cell.cellTypeData.generator.pulseType;
            object->typeData.cell.cellTypeData.generator.alternationInterval = objectTO.typeData.cell.cellTypeData.generator.alternationInterval;
            object->typeData.cell.cellTypeData.generator.numPulses = objectTO.typeData.cell.cellTypeData.generator.numPulses;
        } break;
        case CellType_Attacker: {
            object->typeData.cell.cellTypeData.attacker.mode = objectTO.typeData.cell.cellTypeData.attacker.mode;
            if (objectTO.typeData.cell.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                object->typeData.cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
            } else if (objectTO.typeData.cell.cellTypeData.attacker.mode == AttackerMode_Creature) {
                object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.minNumCells =
                    objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.minNumCells;
                object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.maxNumCells =
                    objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToLineage =
                    objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
            }
        } break;
        case CellType_Injector: {
            object->typeData.cell.cellTypeData.injector.geneIndex = objectTO.typeData.cell.cellTypeData.injector.geneIndex;
        } break;
        case CellType_Muscle: {
            object->typeData.cell.cellTypeData.muscle.mode = objectTO.typeData.cell.cellTypeData.muscle.mode;
            if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle;
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.forward =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.forward;
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.activation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.activation;
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.activationCountdown =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.activationCountdown;
                object->typeData.cell.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                object->typeData.cell.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                object->typeData.cell.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                object->typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle;
                object->typeData.cell.cellTypeData.muscle.modeData.manualBending.lastAngleDelta =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
                object->typeData.cell.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                object->typeData.cell.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                object->typeData.cell.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                object->typeData.cell.cellTypeData.muscle.modeData.angleBending.initialAngle =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.initialAngle;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.initialDistance =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.forward =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.forward;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.activation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.activation;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.activationCountdown =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
                object->typeData.cell.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                object->typeData.cell.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                object->typeData.cell.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                object->typeData.cell.cellTypeData.muscle.modeData.manualCrawling.initialDistance =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.initialDistance;
                object->typeData.cell.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
                object->typeData.cell.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
                object->typeData.cell.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied =
                    objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
            }
            object->typeData.cell.cellTypeData.muscle.lastMovementX = objectTO.typeData.cell.cellTypeData.muscle.lastMovementX;
            object->typeData.cell.cellTypeData.muscle.lastMovementY = objectTO.typeData.cell.cellTypeData.muscle.lastMovementY;
        } break;
        case CellType_Defender: {
            object->typeData.cell.cellTypeData.defender.mode = objectTO.typeData.cell.cellTypeData.defender.mode;
        } break;
        case CellType_Reconnector: {
            object->typeData.cell.cellTypeData.reconnector.mode = objectTO.typeData.cell.cellTypeData.reconnector.mode;
            if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
            } else if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                object->typeData.cell.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor;
            } else if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                object->typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells =
                    objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                object->typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells =
                    objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                object->typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor;
                object->typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage =
                    objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
            }
        } break;
        case CellType_Detonator: {
            object->typeData.cell.cellTypeData.detonator.state = objectTO.typeData.cell.cellTypeData.detonator.state;
            object->typeData.cell.cellTypeData.detonator.countdown = objectTO.typeData.cell.cellTypeData.detonator.countdown;
        } break;
        case CellType_Digestor: {
            object->typeData.cell.cellTypeData.digestor.rawEnergyConductivity = objectTO.typeData.cell.cellTypeData.digestor.rawEnergyConductivity;
        } break;
        case CellType_Memory: {
            object->typeData.cell.cellTypeData.memory.mode = objectTO.typeData.cell.cellTypeData.memory.mode;
            object->typeData.cell.cellTypeData.memory.numSignalEntries = objectTO.typeData.cell.cellTypeData.memory.numSignalEntries;
            object->typeData.cell.cellTypeData.memory.channelBitMask = objectTO.typeData.cell.cellTypeData.memory.channelBitMask;
            if (objectTO.typeData.cell.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                object->typeData.cell.cellTypeData.memory.modeData.signalDelay.delay = objectTO.typeData.cell.cellTypeData.memory.modeData.signalDelay.delay;
                object->typeData.cell.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized;
                object->typeData.cell.cellTypeData.memory.modeData.signalDelay.ringBufferIndex =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalDelay.ringBufferIndex;
            } else if (objectTO.typeData.cell.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                object->typeData.cell.cellTypeData.memory.modeData.signalRecorder.readOnly =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalRecorder.readOnly;
                object->typeData.cell.cellTypeData.memory.modeData.signalRecorder.state =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalRecorder.state;
                object->typeData.cell.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
                object->typeData.cell.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries;
            } else if (objectTO.typeData.cell.cellTypeData.memory.mode == MemoryMode_SignalStorage) {
                object->typeData.cell.cellTypeData.memory.modeData.signalStorage.readOnly =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalStorage.readOnly;
            } else if (objectTO.typeData.cell.cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
                object->typeData.cell.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight =
                    objectTO.typeData.cell.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
            }
            copyDataToHeap(
                sizeof(SignalEntryTO) * objectTO.typeData.cell.cellTypeData.memory.numSignalEntries,
                objectTO.typeData.cell.cellTypeData.memory.signalEntriesDataIndex,
                to.heap,
                reinterpret_cast<uint8_t*&>(object->typeData.cell.cellTypeData.memory.signalEntries));
        } break;
        case CellType_Communicator: {
            object->typeData.cell.cellTypeData.communicator.mode = objectTO.typeData.cell.cellTypeData.communicator.mode;
            if (objectTO.typeData.cell.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                object->typeData.cell.cellTypeData.communicator.modeData.sender.range = objectTO.typeData.cell.cellTypeData.communicator.modeData.sender.range;
                object->typeData.cell.cellTypeData.communicator.modeData.sender.maxTimesSent =
                    objectTO.typeData.cell.cellTypeData.communicator.modeData.sender.maxTimesSent;
            } else if (objectTO.typeData.cell.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                object->typeData.cell.cellTypeData.communicator.modeData.receiver.restrictToColor =
                    objectTO.typeData.cell.cellTypeData.communicator.modeData.receiver.restrictToColor;
                object->typeData.cell.cellTypeData.communicator.modeData.receiver.restrictToLineage =
                    objectTO.typeData.cell.cellTypeData.communicator.modeData.receiver.restrictToLineage;
            }
        } break;
        }
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
    object->stiffness = _data->primaryNumberGen.random();
    object->numConnections = 0;
    object->locked = 0;
    object->selected = 0;
    object->detached = 0;
    object->scheduledOperationIndex = -1;
    object->color = 0;
    object->fixed = false;
    object->sticky = false;
    object->density = 1.0f;
    object->type = ObjectType_FreeCell;
    object->typeData.freeCell.event = CellEvent_No;
    object->typeData.freeCell.rawEnergy = energy;

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
    object->typeData.cell.usableEnergy = energy;
    object->typeData.cell.rawEnergy = 0;
    object->stiffness = gene->stiffness;
    object->color = node->color;
    object->typeData.cell.frontAngle = VALUE_NOT_SET_FLOAT;
    object->stiffness = 1.0f;
    object->fixed = false;
    object->sticky = false;
    object->typeData.cell.age = 0;
    object->typeData.cell.cellState = CellState_Constructing;
    object->typeData.cell.creature = creature;
    object->typeData.cell.nodeIndex = nodeIndex;
    object->typeData.cell.parentNodeIndex = parentNodeIndex;
    object->typeData.cell.geneIndex = geneIndex;
    object->numConnections = 0;
    object->typeData.cell.frontAngleId = 0;
    object->typeData.cell.headCell = false;

    object->typeData.cell.neuralNetwork = _data->entities.heap.getTypedSubArray<NeuralNetwork>(1);
    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        object->typeData.cell.neuralNetwork->weights[i] = node->neuralNetwork.weights[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->typeData.cell.neuralNetwork->biases[i] = node->neuralNetwork.biases[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->typeData.cell.neuralNetwork->activationFunctions[i] = node->neuralNetwork.activationFunctions[i];
    }
    object->typeData.cell.signalRestriction.mode = node->signalRestriction.mode;
    object->typeData.cell.signalRestriction.baseAngle = node->signalRestriction.baseAngle;
    object->typeData.cell.signalRestriction.openingAngle = node->signalRestriction.openingAngle;
    object->typeData.cell.signalState = 0;
    object->typeData.cell.activationTime = 0;
    object->typeData.cell.cellTriggered = CellTriggered_No;
    object->typeData.cell.event = CellEvent_No;
    object->selected = 0;
    object->detached = 0;
    object->locked = 0;
    object->density = 1.0f;
    object->scheduledOperationIndex = -1;

    switch (node->cellType) {
    case CellTypeGenome_Base: {
        object->typeData.cell.cellType = CellType_Base;
    } break;
    case CellTypeGenome_Depot: {
        object->typeData.cell.cellType = CellType_Depot;
        object->typeData.cell.cellTypeData.depot.storageLimit = node->cellTypeData.depot.storageLimit;
        object->typeData.cell.cellTypeData.depot.storedUsableEnergy = node->cellTypeData.depot.initialStoredUsableEnergy;
    } break;
    case CellTypeGenome_Constructor: {
        object->typeData.cell.cellType = CellType_Constructor;
        auto const& nodeConstructor = node->cellTypeData.constructor;
        auto& constructor = object->typeData.cell.cellTypeData.constructor;
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
        object->typeData.cell.cellType = CellType_Sensor;
        auto const& nodeSensor = node->cellTypeData.sensor;
        auto& sensor = object->typeData.cell.cellTypeData.sensor;
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
        object->typeData.cell.cellType = CellType_Generator;
        auto const& nodeGenerator = node->cellTypeData.generator;
        auto& generator = object->typeData.cell.cellTypeData.generator;
        generator.autoTriggerInterval = nodeGenerator.autoTriggerInterval;
        generator.pulseType = nodeGenerator.pulseType;
        generator.alternationInterval = nodeGenerator.alternationInterval;
        generator.numPulses = 0;
    } break;
    case CellTypeGenome_Attacker: {
        object->typeData.cell.cellType = CellType_Attacker;
        auto const& nodeAttacker = node->cellTypeData.attacker;
        auto& attacker = object->typeData.cell.cellTypeData.attacker;
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
        object->typeData.cell.cellType = CellType_Injector;
        auto const& nodeInjector = node->cellTypeData.injector;
        auto& injector = object->typeData.cell.cellTypeData.injector;
        injector.geneIndex = nodeInjector.geneIndex;
    } break;
    case CellTypeGenome_Muscle: {
        object->typeData.cell.cellType = CellType_Muscle;
        auto const& nodeMuscle = node->cellTypeData.muscle;
        auto& muscle = object->typeData.cell.cellTypeData.muscle;
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
        object->typeData.cell.cellType = CellType_Defender;
        auto const& nodeDefender = node->cellTypeData.defender;
        auto& defender = object->typeData.cell.cellTypeData.defender;
        defender.mode = nodeDefender.mode;
    } break;
    case CellTypeGenome_Reconnector: {
        object->typeData.cell.cellType = CellType_Reconnector;
        auto const& nodeReconnector = node->cellTypeData.reconnector;
        auto& reconnector = object->typeData.cell.cellTypeData.reconnector;
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
        object->typeData.cell.cellType = CellType_Detonator;
        auto const& nodeDetonator = node->cellTypeData.detonator;
        auto& detonator = object->typeData.cell.cellTypeData.detonator;
        detonator.state = DetonatorState_Ready;
        detonator.countdown = nodeDetonator.countdown;
    } break;
    case CellTypeGenome_Digestor: {
        object->typeData.cell.cellType = CellType_Digestor;
        auto const& nodeDigestor = node->cellTypeData.digestor;
        auto& digestor = object->typeData.cell.cellTypeData.digestor;
        digestor.rawEnergyConductivity = nodeDigestor.rawEnergyConductivity;
    } break;
    case CellTypeGenome_Memory: {
        object->typeData.cell.cellType = CellType_Memory;
        auto const& nodeMemory = node->cellTypeData.memory;
        auto& memory = object->typeData.cell.cellTypeData.memory;
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
        object->typeData.cell.cellType = CellType_Communicator;
        auto const& nodeCommunicator = node->cellTypeData.communicator;
        auto& communicator = object->typeData.cell.cellTypeData.communicator;
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
