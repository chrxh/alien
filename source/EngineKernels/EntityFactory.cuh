#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>

#include "ConstantMemory.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"
#include "TOs.cuh"

class EntityFactory
{
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Energy* createParticleFromTO(EnergyTO const& particleTO);
    __inline__ __device__ Creature* createCreatureFromTO(TOs const& to, int creatureIndex);
    __inline__ __device__ Genome* createGenomeFromTO(TOs const& to, int genomeIndex);
    __inline__ __device__ Object* createObjectFromTO(TOs const& to, int objectIndex, Object* objectArray);
    __inline__ __device__ void changeObjectFromTO(TOs const& to, ObjectTO const& objectTO, Object* object);
    __inline__ __device__ void changeEnergyFromTO(EnergyTO const& particleTO, Energy* particle);

    __inline__ __device__ Energy* createEnergy(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Object* createFreeCell(float energy, float2 const& pos, float2 const& vel);

    __inline__ __device__ Creature* cloneCreature(Creature* creature);
    __inline__ __device__ Genome* cloneGenome(Genome* genome);

    __inline__ __device__ Object* createCellFromNode(
        uint64_t& objectIndex,
        Creature* creature,
        int geneIndex,
        int nodeIndex,
        bool homogeneousCellType,
        int parentNodeIndex,
        int concatenationIndex,
        int branchIndex,
        float2 pos,
        float2 vel,
        float usableEnergy);

    __inline__ __device__ Genome* createEmptyGenome();
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

__inline__ __device__ Genome* EntityFactory::createGenomeFromTO(TOs const& to, int genomeIndex)
{
    auto& genomeTO = to.genomes[genomeIndex];
    auto genome = _data->entities.heap.getTypedSubArray<Genome>(1);
    genomeTO.genomeIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(genome) - _data->entities.heap.getArray());
    genome->id = genomeTO.id;
    genome->lineageId = genomeTO.lineageId;
    genome->prevLineageId = genomeTO.prevLineageId;
    genome->frontAngle = genomeTO.frontAngle;
    genome->accumulatedMutations = genomeTO.accumulatedMutations;
    for (int i = 0; i < 2; ++i) {
        genome->mutationRates.neuronMutations[i] = {
            genomeTO.mutationRates.neuronMutations[i].eventProbability,
            genomeTO.mutationRates.neuronMutations[i].weightSigma,
            genomeTO.mutationRates.neuronMutations[i].biasSigma,
            genomeTO.mutationRates.neuronMutations[i].activationFunctionProbability};
        genome->mutationRates.connectionMutations[i] = {
            genomeTO.mutationRates.connectionMutations[i].eventProbability, genomeTO.mutationRates.connectionMutations[i].sigma};
        genome->mutationRates.cellTypePropertiesMutations[i] = {
            genomeTO.mutationRates.cellTypePropertiesMutations[i].eventProbability,
            genomeTO.mutationRates.cellTypePropertiesMutations[i].sigma,
            genomeTO.mutationRates.cellTypePropertiesMutations[i].probability};
        genome->mutationRates.constructorMutations[i] = {
            genomeTO.mutationRates.constructorMutations[i].eventProbability,
            genomeTO.mutationRates.constructorMutations[i].sigma,
            genomeTO.mutationRates.constructorMutations[i].probability};
    }
    genome->mutationRates.cellTypeModeMutation = {genomeTO.mutationRates.cellTypeModeMutation.eventProbability};
    genome->mutationRates.cellTypeMutation = {genomeTO.mutationRates.cellTypeMutation.eventProbability};
    genome->mutationRates.voidMutation = {genomeTO.mutationRates.voidMutation.eventProbability};
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
        gene.stiffness = geneTO.stiffness;
        gene.connectionDistance = geneTO.connectionDistance;
        gene.homogeneCellType = geneTO.homogeneCellType;
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
            for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
                node.neuralNetwork.weights[i] = nodeTO.neuralNetwork.weights[i];
            }
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                node.neuralNetwork.biases[i] = nodeTO.neuralNetwork.biases[i];
                node.neuralNetwork.activationFunctions[i] = nodeTO.neuralNetwork.activationFunctions[i];
            }
            for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
                node.neuralNetwork.connectionWeights[i] = nodeTO.neuralNetwork.connectionWeights[i];
            }

            node.cellType = nodeTO.cellType;

            switch (nodeTO.cellType) {
            case CellType_Base:
                break;
            case CellType_Depot:
                node.cellTypeData.depot.storageLimit = nodeTO.cellTypeData.depot.storageLimit;
                node.cellTypeData.depot.initialStoredUsableEnergy = nodeTO.cellTypeData.depot.initialStoredUsableEnergy;
                break;
            case CellType_Sensor:
                node.cellTypeData.sensor.autoTrigger = nodeTO.cellTypeData.sensor.autoTrigger;
                node.cellTypeData.sensor.tagForAttackers = nodeTO.cellTypeData.sensor.tagForAttackers;
                node.cellTypeData.sensor.minRange = nodeTO.cellTypeData.sensor.minRange;
                node.cellTypeData.sensor.maxRange = nodeTO.cellTypeData.sensor.maxRange;
                node.cellTypeData.sensor.mode = nodeTO.cellTypeData.sensor.mode;
                if (nodeTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                    node.cellTypeData.sensor.modeData.detectEnergy.minDensity = nodeTO.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectSolid) {
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                    node.cellTypeData.sensor.modeData.detectFreeCell.minDensity = nodeTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                    node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors = nodeTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors;
                } else if (nodeTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                    node.cellTypeData.sensor.modeData.detectCreature.minNumCells = nodeTO.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                    node.cellTypeData.sensor.modeData.detectCreature.maxNumCells = nodeTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                    node.cellTypeData.sensor.modeData.detectCreature.restrictToColors = nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToColors;
                    node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage = nodeTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                }
                break;
            case CellType_Generator:
                node.cellTypeData.generator.additive = nodeTO.cellTypeData.generator.additive;
                node.cellTypeData.generator.minValue = nodeTO.cellTypeData.generator.minValue;
                node.cellTypeData.generator.maxValue = nodeTO.cellTypeData.generator.maxValue;
                node.cellTypeData.generator.timeOffset = nodeTO.cellTypeData.generator.timeOffset;
                node.cellTypeData.generator.mode = nodeTO.cellTypeData.generator.mode;
                if (nodeTO.cellTypeData.generator.mode == GeneratorMode_SquareSignal) {
                    node.cellTypeData.generator.modeData.squareSignal.period = nodeTO.cellTypeData.generator.modeData.squareSignal.period;
                } else if (nodeTO.cellTypeData.generator.mode == GeneratorMode_SawtoothSignal) {
                    node.cellTypeData.generator.modeData.sawtoothSignal.period = nodeTO.cellTypeData.generator.modeData.sawtoothSignal.period;
                }
                break;
            case CellType_Attacker:
                node.cellTypeData.attacker.mode = nodeTO.cellTypeData.attacker.mode;
                if (nodeTO.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                    node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors = nodeTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors;
                }
                break;
            case CellType_Injector:
                node.cellTypeData.injector.geneIndex = nodeTO.cellTypeData.injector.geneIndex;
                break;
            case CellType_Muscle:
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
            case CellType_Defender:
                node.cellTypeData.defender.mode = nodeTO.cellTypeData.defender.mode;
                break;
            case CellType_Reconnector:
                node.cellTypeData.reconnector.mode = nodeTO.cellTypeData.reconnector.mode;
                if (nodeTO.cellTypeData.reconnector.mode == ReconnectorMode_Solid) {
                } else if (nodeTO.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                    node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors =
                        nodeTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors;
                } else if (nodeTO.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                    node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells =
                        nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                    node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells =
                        nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                    node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors =
                        nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors;
                    node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage =
                        nodeTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
                }
                break;
            case CellType_Detonator:
                node.cellTypeData.detonator.countdown = nodeTO.cellTypeData.detonator.countdown;
                break;
            case CellType_Digestor:
                node.cellTypeData.digestor.rawEnergyConductivity = nodeTO.cellTypeData.digestor.rawEnergyConductivity;
                break;
            case CellType_Memory: {
                node.cellTypeData.memory.mode = nodeTO.cellTypeData.memory.mode;
                node.cellTypeData.memory.numSignalEntries = nodeTO.cellTypeData.memory.numSignalEntries;
                node.cellTypeData.memory.channelBitMask = nodeTO.cellTypeData.memory.channelBitMask;
                if (nodeTO.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                    node.cellTypeData.memory.modeData.signalDelay.delay = nodeTO.cellTypeData.memory.modeData.signalDelay.delay;
                } else if (nodeTO.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                    node.cellTypeData.memory.modeData.signalRecorder.readOnly = nodeTO.cellTypeData.memory.modeData.signalRecorder.readOnly;
                    node.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries =
                        nodeTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
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
                    for (int l = 0; l < NEURONS_PER_CELL; ++l) {
                        signalEntries[k].channels[l] = entriesTO[k].channels[l];
                    }
                }
            } break;
            case CellType_Communicator:
                node.cellTypeData.communicator.mode = nodeTO.cellTypeData.communicator.mode;
                if (nodeTO.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                    node.cellTypeData.communicator.modeData.sender.range = nodeTO.cellTypeData.communicator.modeData.sender.range;
                    node.cellTypeData.communicator.modeData.sender.maxTimesSent = nodeTO.cellTypeData.communicator.modeData.sender.maxTimesSent;
                } else if (nodeTO.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                    node.cellTypeData.communicator.modeData.receiver.restrictToColors = nodeTO.cellTypeData.communicator.modeData.receiver.restrictToColors;
                    node.cellTypeData.communicator.modeData.receiver.restrictToLineage = nodeTO.cellTypeData.communicator.modeData.receiver.restrictToLineage;
                }
                break;
            case CellType_Void:
                break;
            }

            // Handle optional constructor field
            node.constructorAvailable = nodeTO.constructorAvailable;
            if (nodeTO.constructorAvailable) {
                node.constructor.autoTriggerInterval = nodeTO.constructor.autoTriggerInterval;
                node.constructor.geneIndex = nodeTO.constructor.geneIndex;
                node.constructor.constructionActivationTime = nodeTO.constructor.constructionActivationTime;
                node.constructor.constructionAngle = nodeTO.constructor.constructionAngle;
                node.constructor.provideEnergy = nodeTO.constructor.provideEnergy;
                node.constructor.reservedEnergy = nodeTO.constructor.reservedEnergy;
                node.constructor.separation = nodeTO.constructor.separation;
                node.constructor.numBranches = nodeTO.constructor.numBranches;
                node.constructor.numConcatenations = nodeTO.constructor.numConcatenations;
            }
        }
    }
    return genome;
}

__inline__ __device__ Creature* EntityFactory::createCreatureFromTO(TOs const& to, int creatureIndex)
{
    auto& creatureTO = to.creatures[creatureIndex];
    auto creature = _data->entities.heap.getTypedSubArray<Creature>(1);
    creatureTO.creatureIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(creature) - _data->entities.heap.getArray());

    creature->id = creatureTO.id;
    creature->ancestorId = creatureTO.ancestorId;
    creature->generation = creatureTO.generation;
    creature->numCells = creatureTO.numCells;
    creature->mutationState = creatureTO.mutationState;
    creature->headUpdateId = creatureTO.headUpdateId;

    auto const& genomeTO = to.genomes[creatureTO.genomeArrayIndex];
    creature->genome = &_data->entities.heap.atType<Genome>(genomeTO.genomeIndexOnGpu);

    return creature;
}

__inline__ __device__ Object* EntityFactory::createObjectFromTO(TOs const& to, int objectIndex, Object* objectArray)
{
    auto objectTO = to.objects[objectIndex];
    Object** objectPointer = _data->entities.objects.getNewElement();
    Object* object = objectArray + objectIndex;
    *objectPointer = object;

    object->flags = 0;
    changeObjectFromTO(to, objectTO, object);
    object->id = objectTO.id;
    object->locked = 0;
    object->selected = 0;
    object->numConnections = objectTO.numConnections;
    if (object->type == ObjectType_Cell) {
        auto const& genomeTO = to.creatures[objectTO.typeData.cell.creatureIndex];
        object->typeData.cell.creature = &_data->entities.heap.atType<Creature>(genomeTO.creatureIndexOnGpu);
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

__inline__ __device__ void EntityFactory::changeObjectFromTO(TOs const& to, ObjectTO const& objectTO, Object* object)
{
    object->id = objectTO.id;
    object->pos = objectTO.pos;
    _map.correctPosition(object->pos);
    object->vel = objectTO.vel;
    object->stiffness = objectTO.stiffness;
    object->color = objectTO.color;
    object->setFixed(objectTO.isFixed());
    object->setSticky(objectTO.isSticky());
    object->type = objectTO.type;

    if (objectTO.type == ObjectType_Solid) {
        object->typeData.solid.energy = objectTO.typeData.solid.energy;
    } else if (objectTO.type == ObjectType_Fluid) {
        object->typeData.fluid.energy = objectTO.typeData.fluid.energy;
        object->typeData.fluid.glow = objectTO.typeData.fluid.glow;
    } else if (objectTO.type == ObjectType_FreeCell) {
        object->typeData.freeCell.energy = objectTO.typeData.freeCell.energy;
        object->typeData.freeCell.age = objectTO.typeData.freeCell.age;
        object->typeData.freeCell.event = CellEvent_No;
        object->typeData.freeCell.eventCounter = 0;
        object->typeData.freeCell.eventPos = {0, 0};
    } else if (objectTO.type == ObjectType_Cell) {
        auto const& cell = &object->typeData.cell;
        auto const& cellTO = objectTO.typeData.cell;

        cell->cellState = cellTO.cellState;
        cell->usableEnergy = cellTO.usableEnergy;
        cell->rawEnergy = cellTO.rawEnergy;
        cell->cellType = cellTO.cellType;
        cell->age = cellTO.age;
        cell->frontAngle = cellTO.frontAngle;
        cell->activationTime = cellTO.activationTime;
        cell->lastUpdate = cellTO.lastUpdate;
        cell->concatenationIndex = cellTO.concatenationIndex;
        cell->branchIndex = cellTO.branchIndex;
        cell->nodeIndex = cellTO.nodeIndex;
        cell->parentNodeIndex = cellTO.parentNodeIndex;
        cell->geneIndex = cellTO.geneIndex;
        cell->headUpdateId = cellTO.headUpdateId;
        cell->headCell = cellTO.headCell;
        cell->event = cellTO.event;
        cell->eventCounter = cellTO.eventCounter;
        cell->eventPos = cellTO.eventPos;

        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            cell->signal.channels[i] = cellTO.signal.channels[i];
        }
        cell->signal.numTimesSent = cellTO.signal.numTimesSent;
        cell->signalChanges = 0;

        cell->cellType = cellTO.cellType;

        // Copy NeuralNetTO to NeuralNet
        {
            auto* nnTO = reinterpret_cast<NeuralNetTO*>(&to.heap[cellTO.neuralNetworkDataIndex]);
            cell->neuralNetwork = _data->entities.heap.getTypedSubArray<NeuralNet>(1);
            for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
                cell->neuralNetwork->weights[i] = nnTO->weights[i];
            }
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                cell->neuralNetwork->biases[i] = nnTO->biases[i];
                cell->neuralNetwork->activationFunctions[i] = nnTO->activationFunctions[i];
            }
            for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
                cell->neuralNetwork->connectionWeights[i] = nnTO->connectionWeights[i];
            }
        }

        switch (cellTO.cellType) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            cell->cellTypeData.depot.storageLimit = cellTO.cellTypeData.depot.storageLimit;
            cell->cellTypeData.depot.storedUsableEnergy = cellTO.cellTypeData.depot.storedUsableEnergy;
        } break;
        case CellType_Sensor: {
            cell->cellTypeData.sensor.autoTrigger = cellTO.cellTypeData.sensor.autoTrigger;
            cell->cellTypeData.sensor.tagForAttackers = cellTO.cellTypeData.sensor.tagForAttackers;
            cell->cellTypeData.sensor.minRange = cellTO.cellTypeData.sensor.minRange;
            cell->cellTypeData.sensor.maxRange = cellTO.cellTypeData.sensor.maxRange;
            cell->cellTypeData.sensor.mode = cellTO.cellTypeData.sensor.mode;
            if (cellTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                cell->cellTypeData.sensor.modeData.detectEnergy.minDensity = cellTO.cellTypeData.sensor.modeData.detectEnergy.minDensity;
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectSolid) {
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                cell->cellTypeData.sensor.modeData.detectFreeCell.minDensity = cellTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                cell->cellTypeData.sensor.modeData.detectFreeCell.restrictToColors = cellTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors;
            } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                cell->cellTypeData.sensor.modeData.detectCreature.minNumCells = cellTO.cellTypeData.sensor.modeData.detectCreature.minNumCells;
                cell->cellTypeData.sensor.modeData.detectCreature.maxNumCells = cellTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
                cell->cellTypeData.sensor.modeData.detectCreature.restrictToColors = cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToColors;
                cell->cellTypeData.sensor.modeData.detectCreature.restrictToLineage = cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            }
            cell->cellTypeData.sensor.lastMatchAvailable = cellTO.cellTypeData.sensor.lastMatchAvailable;
            cell->cellTypeData.sensor.lastMatch.creatureIdPart = cellTO.cellTypeData.sensor.lastMatch.creatureIdPart;
            cell->cellTypeData.sensor.lastMatch.pos = cellTO.cellTypeData.sensor.lastMatch.pos;
        } break;
        case CellType_Generator: {
            cell->cellTypeData.generator.additive = cellTO.cellTypeData.generator.additive;
            cell->cellTypeData.generator.minValue = cellTO.cellTypeData.generator.minValue;
            cell->cellTypeData.generator.maxValue = cellTO.cellTypeData.generator.maxValue;
            cell->cellTypeData.generator.timeOffset = cellTO.cellTypeData.generator.timeOffset;
            cell->cellTypeData.generator.mode = cellTO.cellTypeData.generator.mode;
            if (cellTO.cellTypeData.generator.mode == GeneratorMode_SquareSignal) {
                cell->cellTypeData.generator.modeData.squareSignal.period = cellTO.cellTypeData.generator.modeData.squareSignal.period;
            } else if (cellTO.cellTypeData.generator.mode == GeneratorMode_SawtoothSignal) {
                cell->cellTypeData.generator.modeData.sawtoothSignal.period = cellTO.cellTypeData.generator.modeData.sawtoothSignal.period;
            }
            cell->cellTypeData.generator.numPulses = cellTO.cellTypeData.generator.numPulses;
        } break;
        case CellType_Attacker: {
            cell->cellTypeData.attacker.mode = cellTO.cellTypeData.attacker.mode;
            if (cellTO.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                cell->cellTypeData.attacker.modeData.attackFreeCell.restrictToColors = cellTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors;
            }
        } break;
        case CellType_Injector: {
            cell->cellTypeData.injector.geneIndex = cellTO.cellTypeData.injector.geneIndex;
        } break;
        case CellType_Muscle: {
            cell->cellTypeData.muscle.mode = cellTO.cellTypeData.muscle.mode;
            if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
                cell->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                cell->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                cell->cellTypeData.muscle.modeData.autoBending.initialAngle = cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle;
                cell->cellTypeData.muscle.modeData.autoBending.forward = cellTO.cellTypeData.muscle.modeData.autoBending.forward;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                cell->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                cell->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                cell->cellTypeData.muscle.modeData.manualBending.initialAngle = cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle;
                cell->cellTypeData.muscle.modeData.manualBending.lastAngleDelta = cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                cell->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                cell->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio =
                    cellTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                cell->cellTypeData.muscle.modeData.angleBending.initialAngle = cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                cell->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                cell->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                cell->cellTypeData.muscle.modeData.autoCrawling.initialDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
                cell->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                cell->cellTypeData.muscle.modeData.autoCrawling.forward = cellTO.cellTypeData.muscle.modeData.autoCrawling.forward;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                cell->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                cell->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio =
                    cellTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                cell->cellTypeData.muscle.modeData.manualCrawling.initialDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance;
                cell->cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
                cell->cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = cellTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
            } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
            }
            cell->cellTypeData.muscle.lastMovementX = cellTO.cellTypeData.muscle.lastMovementX;
            cell->cellTypeData.muscle.lastMovementY = cellTO.cellTypeData.muscle.lastMovementY;
        } break;
        case CellType_Defender: {
            cell->cellTypeData.defender.mode = cellTO.cellTypeData.defender.mode;
        } break;
        case CellType_Reconnector: {
            cell->cellTypeData.reconnector.mode = cellTO.cellTypeData.reconnector.mode;
            if (cellTO.cellTypeData.reconnector.mode == ReconnectorMode_Solid) {
            } else if (cellTO.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                cell->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors =
                    cellTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors;
            } else if (cellTO.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                cell->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells = cellTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells;
                cell->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells = cellTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells;
                cell->cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors =
                    cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors;
                cell->cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage =
                    cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
            }
        } break;
        case CellType_Detonator: {
            cell->cellTypeData.detonator.state = cellTO.cellTypeData.detonator.state;
            cell->cellTypeData.detonator.countdown = cellTO.cellTypeData.detonator.countdown;
        } break;
        case CellType_Digestor: {
            cell->cellTypeData.digestor.rawEnergyConductivity = cellTO.cellTypeData.digestor.rawEnergyConductivity;
        } break;
        case CellType_Memory: {
            cell->cellTypeData.memory.mode = cellTO.cellTypeData.memory.mode;
            cell->cellTypeData.memory.numSignalEntries = cellTO.cellTypeData.memory.numSignalEntries;
            cell->cellTypeData.memory.channelBitMask = cellTO.cellTypeData.memory.channelBitMask;
            if (cellTO.cellTypeData.memory.mode == MemoryMode_SignalDelay) {
                cell->cellTypeData.memory.modeData.signalDelay.delay = cellTO.cellTypeData.memory.modeData.signalDelay.delay;
                cell->cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized =
                    cellTO.cellTypeData.memory.modeData.signalDelay.numSignalEntriesInitialized;
                cell->cellTypeData.memory.modeData.signalDelay.ringBufferIndex = cellTO.cellTypeData.memory.modeData.signalDelay.ringBufferIndex;
            } else if (cellTO.cellTypeData.memory.mode == MemoryMode_SignalRecorder) {
                cell->cellTypeData.memory.modeData.signalRecorder.readOnly = cellTO.cellTypeData.memory.modeData.signalRecorder.readOnly;
                cell->cellTypeData.memory.modeData.signalRecorder.state = cellTO.cellTypeData.memory.modeData.signalRecorder.state;
                cell->cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries =
                    cellTO.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries;
                cell->cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries =
                    cellTO.cellTypeData.memory.modeData.signalRecorder.numReadSignalEntries;
            } else if (cellTO.cellTypeData.memory.mode == MemoryMode_SignalStorage) {
                cell->cellTypeData.memory.modeData.signalStorage.readOnly = cellTO.cellTypeData.memory.modeData.signalStorage.readOnly;
            } else if (cellTO.cellTypeData.memory.mode == MemoryMode_SignalIntegrator) {
                cell->cellTypeData.memory.modeData.signalIntegrator.newSignalWeight = cellTO.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight;
            }
            copyDataToHeap(
                sizeof(SignalEntryTO) * cellTO.cellTypeData.memory.numSignalEntries,
                cellTO.cellTypeData.memory.signalEntriesDataIndex,
                to.heap,
                reinterpret_cast<uint8_t*&>(cell->cellTypeData.memory.signalEntries));
        } break;
        case CellType_Communicator: {
            cell->cellTypeData.communicator.mode = cellTO.cellTypeData.communicator.mode;
            if (cellTO.cellTypeData.communicator.mode == CommunicatorMode_Sender) {
                cell->cellTypeData.communicator.modeData.sender.range = cellTO.cellTypeData.communicator.modeData.sender.range;
                cell->cellTypeData.communicator.modeData.sender.maxTimesSent = cellTO.cellTypeData.communicator.modeData.sender.maxTimesSent;
            } else if (cellTO.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
                cell->cellTypeData.communicator.modeData.receiver.restrictToColors = cellTO.cellTypeData.communicator.modeData.receiver.restrictToColors;
                cell->cellTypeData.communicator.modeData.receiver.restrictToLineage = cellTO.cellTypeData.communicator.modeData.receiver.restrictToLineage;
            }
        } break;
        case CellType_Void: {
        } break;
        }

        cell->constructorAvailable = cellTO.constructorAvailable;
        if (cellTO.constructorAvailable) {
            cell->constructor.autoTriggerInterval = cellTO.constructor.autoTriggerInterval;
            cell->constructor.constructionActivationTime = cellTO.constructor.constructionActivationTime;
            cell->constructor.constructionAngle = cellTO.constructor.constructionAngle;
            cell->constructor.provideEnergy = cellTO.constructor.provideEnergy;
            cell->constructor.reservedEnergy = cellTO.constructor.reservedEnergy;
            cell->constructor.separation = cellTO.constructor.separation;
            cell->constructor.numBranches = cellTO.constructor.numBranches;
            cell->constructor.numConcatenations = cellTO.constructor.numConcatenations;
            cell->constructor.geneIndex = cellTO.constructor.geneIndex;
            cell->constructor.lastConstructedCellId = cellTO.constructor.lastConstructedCellId;
            cell->constructor.currentOffspring = cellTO.constructor.currentOffspring;
            cell->constructor.offspring = nullptr;
            cell->constructor.energyNeeded = false;
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
    particle->id = _data->primaryNumberGen.createEntityId();
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

    object->id = _data->primaryNumberGen.createEntityId();
    object->pos = pos;
    object->vel = vel;
    object->stiffness = _data->primaryNumberGen.random();
    object->numConnections = 0;
    object->locked = 0;
    object->selected = 0;
    object->flags = 0;
    object->color = 0;
    object->density = 1.0f;
    object->type = ObjectType_FreeCell;
    object->typeData.freeCell.event = CellEvent_No;
    object->typeData.freeCell.energy = energy;
    object->typeData.freeCell.age = 0;

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
    newCreature->mutationState = MutationState_NotMutated;
    return newCreature;
}

__inline__ __device__ Genome* EntityFactory::cloneGenome(Genome* genome)
{
    auto newGenome = createEmptyGenome();
    auto newId = newGenome->id;
    *newGenome = *genome;
    newGenome->id = newId;

    auto newGenes = createEmptyGenes(genome->numGenes);
    for (int i = 0, numGenes = genome->numGenes; i < numGenes; ++i) {
        auto gene = &genome->genes[i];
        newGenes[i] = *gene;
        auto newNodes = createEmptyNodes(gene->numNodes);
        for (int j = 0, numNodes = gene->numNodes; j < numNodes; ++j) {
            auto node = &gene->nodes[j];
            newNodes[j] = *node;
        }
        newGenes[i].nodes = newNodes;
    }
    newGenome->genes = newGenes;

    return newGenome;
}

__inline__ __device__ Object* EntityFactory::createCellFromNode(
    uint64_t& objectIndex,
    Creature* creature,
    int geneIndex,
    int nodeIndex,
    bool homogeneousCellType,
    int parentNodeIndex,
    int concatenationIndex,
    int branchIndex,
    float2 pos,
    float2 vel,
    float usableEnergy)
{
    auto const& gene = &creature->genome->genes[geneIndex];
    auto const& node = &gene->nodes[nodeIndex];

    // With homogeneous cell type the cell type and its properties are taken from the gene's first node
    auto const& cellTypeNode = homogeneousCellType ? &gene->nodes[0] : node;

    auto object = _data->entities.heap.getTypedSubArray<Object>(1);
    auto objectPointer = _data->entities.objects.getNewElement(&objectIndex);
    *objectPointer = object;
    object->id = _data->primaryNumberGen.createEntityId();
    object->pos = pos;
    object->vel = vel;
    object->stiffness = gene->stiffness;
    object->color = node->color;
    object->flags = 0;
    object->numConnections = 0;
    object->type = ObjectType_Cell;
    object->selected = 0;
    object->locked = 0;
    object->density = 1.0f;

    auto& cell = object->typeData.cell;
    cell.frontAngle = VALUE_NOT_SET_FLOAT;
    cell.usableEnergy = usableEnergy;
    cell.rawEnergy = 0.0f;
    cell.age = 0;
    cell.cellState = CellState_Constructing;
    cell.creature = creature;
    cell.nodeIndex = nodeIndex;
    cell.parentNodeIndex = parentNodeIndex;
    cell.geneIndex = geneIndex;
    cell.concatenationIndex = concatenationIndex;
    cell.branchIndex = branchIndex;
    cell.headUpdateId = 0;
    cell.headCell = false;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        cell.signal.channels[i] = 0.0f;
    }
    cell.signal.numTimesSent = 0;
    cell.signalChanges = 0;

    cell.neuralNetwork = _data->entities.heap.getTypedSubArray<NeuralNet>(1);
    for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
        cell.neuralNetwork->weights[i] = node->neuralNetwork.weights[i];
    }
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        cell.neuralNetwork->biases[i] = node->neuralNetwork.biases[i];
    }
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        cell.neuralNetwork->activationFunctions[i] = node->neuralNetwork.activationFunctions[i];
    }
    for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
        cell.neuralNetwork->connectionWeights[i] = node->neuralNetwork.connectionWeights[i];
    }
    cell.activationTime = 0;
    cell.lastUpdate = 0;
    cell.event = CellEvent_No;

    switch (cellTypeNode->cellType) {
    case CellType_Base: {
        cell.cellType = CellType_Base;
    } break;
    case CellType_Depot: {
        cell.cellType = CellType_Depot;
        cell.cellTypeData.depot.storageLimit = cellTypeNode->cellTypeData.depot.storageLimit;
        cell.cellTypeData.depot.storedUsableEnergy = cellTypeNode->cellTypeData.depot.initialStoredUsableEnergy;
    } break;
    case CellType_Sensor: {
        cell.cellType = CellType_Sensor;
        auto const& nodeSensor = cellTypeNode->cellTypeData.sensor;
        auto& sensor = cell.cellTypeData.sensor;
        sensor.autoTrigger = nodeSensor.autoTrigger;
        sensor.tagForAttackers = nodeSensor.tagForAttackers;
        sensor.minRange = nodeSensor.minRange;
        sensor.maxRange = nodeSensor.maxRange;
        sensor.mode = nodeSensor.mode;
        if (nodeSensor.mode == SensorMode_Telemetry) {
        } else if (nodeSensor.mode == SensorMode_DetectEnergy) {
            sensor.modeData.detectEnergy.minDensity = nodeSensor.modeData.detectEnergy.minDensity;
        } else if (nodeSensor.mode == SensorMode_DetectSolid) {
        } else if (nodeSensor.mode == SensorMode_DetectFreeCell) {
            sensor.modeData.detectFreeCell.minDensity = nodeSensor.modeData.detectFreeCell.minDensity;
            sensor.modeData.detectFreeCell.restrictToColors = nodeSensor.modeData.detectFreeCell.restrictToColors;
        } else if (nodeSensor.mode == SensorMode_DetectCreature) {
            sensor.modeData.detectCreature.minNumCells = nodeSensor.modeData.detectCreature.minNumCells;
            sensor.modeData.detectCreature.maxNumCells = nodeSensor.modeData.detectCreature.maxNumCells;
            sensor.modeData.detectCreature.restrictToColors = nodeSensor.modeData.detectCreature.restrictToColors;
            sensor.modeData.detectCreature.restrictToLineage = nodeSensor.modeData.detectCreature.restrictToLineage;
        }
        sensor.lastMatchAvailable = false;
    } break;
    case CellType_Generator: {
        cell.cellType = CellType_Generator;
        auto const& nodeGenerator = cellTypeNode->cellTypeData.generator;
        auto& generator = cell.cellTypeData.generator;
        generator.additive = nodeGenerator.additive;
        generator.minValue = nodeGenerator.minValue;
        generator.maxValue = nodeGenerator.maxValue;
        generator.timeOffset = nodeGenerator.timeOffset;
        generator.mode = nodeGenerator.mode;
        if (nodeGenerator.mode == GeneratorMode_SquareSignal) {
            generator.modeData.squareSignal.period = nodeGenerator.modeData.squareSignal.period;
        } else if (nodeGenerator.mode == GeneratorMode_SawtoothSignal) {
            generator.modeData.sawtoothSignal.period = nodeGenerator.modeData.sawtoothSignal.period;
        }
        generator.numPulses = 0;
    } break;
    case CellType_Attacker: {
        cell.cellType = CellType_Attacker;
        auto const& nodeAttacker = cellTypeNode->cellTypeData.attacker;
        auto& attacker = cell.cellTypeData.attacker;
        attacker.mode = nodeAttacker.mode;
        if (nodeAttacker.mode == AttackerMode_FreeCell) {
            attacker.modeData.attackFreeCell.restrictToColors = nodeAttacker.modeData.attackFreeCell.restrictToColors;
        }
    } break;
    case CellType_Injector: {
        cell.cellType = CellType_Injector;
        auto const& nodeInjector = cellTypeNode->cellTypeData.injector;
        auto& injector = cell.cellTypeData.injector;
        injector.geneIndex = nodeInjector.geneIndex;
    } break;
    case CellType_Muscle: {
        cell.cellType = CellType_Muscle;
        auto const& nodeMuscle = cellTypeNode->cellTypeData.muscle;
        auto& muscle = cell.cellTypeData.muscle;
        muscle.lastMovementX = 0;
        muscle.lastMovementY = 0;
        muscle.mode = nodeMuscle.mode;
        switch (nodeMuscle.mode) {
        case MuscleMode_AutoBending: {
            muscle.modeData.autoBending.maxAngleDeviation = nodeMuscle.modeData.autoBending.maxAngleDeviation;
            muscle.modeData.autoBending.forwardBackwardRatio = nodeMuscle.modeData.autoBending.forwardBackwardRatio;
            muscle.modeData.autoBending.initialAngle = VALUE_NOT_SET_FLOAT;
            muscle.modeData.autoBending.forward = true;
        } break;
        case MuscleMode_ManualBending: {
            muscle.modeData.manualBending.maxAngleDeviation = nodeMuscle.modeData.manualBending.maxAngleDeviation;
            muscle.modeData.manualBending.forwardBackwardRatio = nodeMuscle.modeData.manualBending.forwardBackwardRatio;
            muscle.modeData.manualBending.initialAngle = VALUE_NOT_SET_FLOAT;
            muscle.modeData.manualBending.lastAngleDelta = 0;
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
        } break;
        case MuscleMode_ManualCrawling: {
            muscle.modeData.manualCrawling.maxDistanceDeviation = nodeMuscle.modeData.manualCrawling.maxDistanceDeviation;
            muscle.modeData.manualCrawling.forwardBackwardRatio = nodeMuscle.modeData.manualCrawling.forwardBackwardRatio;
            muscle.modeData.manualCrawling.initialDistance = VALUE_NOT_SET_FLOAT;
            muscle.modeData.manualCrawling.lastActualDistance = 0;
            muscle.modeData.manualCrawling.lastDistanceDelta = 0;
        } break;
        case MuscleMode_DirectMovement: {
        } break;
        }
    } break;
    case CellType_Defender: {
        cell.cellType = CellType_Defender;
        auto const& nodeDefender = cellTypeNode->cellTypeData.defender;
        auto& defender = cell.cellTypeData.defender;
        defender.mode = nodeDefender.mode;
    } break;
    case CellType_Reconnector: {
        cell.cellType = CellType_Reconnector;
        auto const& nodeReconnector = cellTypeNode->cellTypeData.reconnector;
        auto& reconnector = cell.cellTypeData.reconnector;
        reconnector.mode = nodeReconnector.mode;
        if (nodeReconnector.mode == ReconnectorMode_Solid) {
        } else if (nodeReconnector.mode == ReconnectorMode_FreeCell) {
            reconnector.modeData.reconnectFreeCell.restrictToColors = nodeReconnector.modeData.reconnectFreeCell.restrictToColors;
        } else if (nodeReconnector.mode == ReconnectorMode_Creature) {
            reconnector.modeData.reconnectCreature.minNumCells = nodeReconnector.modeData.reconnectCreature.minNumCells;
            reconnector.modeData.reconnectCreature.maxNumCells = nodeReconnector.modeData.reconnectCreature.maxNumCells;
            reconnector.modeData.reconnectCreature.restrictToColors = nodeReconnector.modeData.reconnectCreature.restrictToColors;
            reconnector.modeData.reconnectCreature.restrictToLineage = nodeReconnector.modeData.reconnectCreature.restrictToLineage;
        }
    } break;
    case CellType_Detonator: {
        cell.cellType = CellType_Detonator;
        auto const& nodeDetonator = cellTypeNode->cellTypeData.detonator;
        auto& detonator = cell.cellTypeData.detonator;
        detonator.state = DetonatorState_Ready;
        detonator.countdown = nodeDetonator.countdown;
    } break;
    case CellType_Digestor: {
        cell.cellType = CellType_Digestor;
        auto const& nodeDigestor = cellTypeNode->cellTypeData.digestor;
        auto& digestor = cell.cellTypeData.digestor;
        digestor.rawEnergyConductivity = nodeDigestor.rawEnergyConductivity;
    } break;
    case CellType_Memory: {
        cell.cellType = CellType_Memory;
        auto const& nodeMemory = cellTypeNode->cellTypeData.memory;
        auto& memory = cell.cellTypeData.memory;
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
            for (int k = 0; k < NEURONS_PER_CELL; ++k) {
                memory.signalEntries[i].channels[k] = nodeMemory.signalEntries[i].channels[k];
            }
        }
    } break;
    case CellType_Communicator: {
        cell.cellType = CellType_Communicator;
        auto const& nodeCommunicator = cellTypeNode->cellTypeData.communicator;
        auto& communicator = cell.cellTypeData.communicator;
        communicator.mode = nodeCommunicator.mode;
        if (nodeCommunicator.mode == CommunicatorMode_Sender) {
            communicator.modeData.sender.range = nodeCommunicator.modeData.sender.range;
            communicator.modeData.sender.maxTimesSent = nodeCommunicator.modeData.sender.maxTimesSent;
        } else if (nodeCommunicator.mode == CommunicatorMode_Receiver) {
            communicator.modeData.receiver.restrictToColors = nodeCommunicator.modeData.receiver.restrictToColors;
            communicator.modeData.receiver.restrictToLineage = nodeCommunicator.modeData.receiver.restrictToLineage;
        }
    } break;
    case CellType_Void: {
        cell.cellType = CellType_Void;
    } break;
    }

    cell.constructorAvailable = node->constructorAvailable;
    if (node->constructorAvailable) {
        auto const& nodeConstructor = node->constructor;
        auto& constructor = cell.constructor;
        constructor.autoTriggerInterval = nodeConstructor.autoTriggerInterval;
        constructor.constructionActivationTime = nodeConstructor.constructionActivationTime;
        constructor.constructionAngle = nodeConstructor.constructionAngle;
        constructor.provideEnergy = ProvideEnergy_ReduceCellEnergy;
        constructor.reservedEnergy = nodeConstructor.reservedEnergy;
        constructor.separation = nodeConstructor.separation;
        constructor.numBranches = nodeConstructor.numBranches;
        constructor.numConcatenations = nodeConstructor.numConcatenations;
        constructor.geneIndex = nodeConstructor.geneIndex;
        constructor.lastConstructedCellId = VALUE_NOT_SET_UINT64;
        constructor.currentOffspring = 0;
        constructor.offspring = nullptr;
        constructor.energyNeeded = false;
    }

    return object;
}

__inline__ __device__ Genome* EntityFactory::createEmptyGenome()
{
    auto genome = _data->entities.heap.getTypedSubArray<Genome>(1);
    genome->id = _data->primaryNumberGen.createEntityId();
    return genome;
}

__inline__ __device__ Creature* EntityFactory::createEmptyCreature()
{
    auto creature = _data->entities.heap.getTypedSubArray<Creature>(1);
    creature->id = _data->primaryNumberGen.createEntityId();
    return creature;
}

__inline__ __device__ Gene* EntityFactory::createEmptyGenes(int numGenes)
{
    return _data->entities.heap.getTypedSubArray<Gene>(numGenes);
}

__inline__ __device__ Node* EntityFactory::createEmptyNodes(int numNodes)
{
    return _data->entities.heap.getTypedSubArray<Node>(numNodes);
}
