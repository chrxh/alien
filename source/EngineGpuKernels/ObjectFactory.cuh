#pragma once

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "TO.cuh"
#include "Map.cuh"
#include "Object.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"

class ObjectFactory
{
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Particle* createParticleFromTO(ParticleTO const& particleTO);
    __inline__ __device__ Creature* createCreatureFromTO(TO const& to, int creatureIndex);
    __inline__ __device__ Genome* createGenomeFromTO(TO const& to, int genomeIndex);
    __inline__ __device__ Cell* createCellFromTO(TO const& to, int cellIndex, Cell* cellArray);
    __inline__ __device__ void changeCellFromTO(TO const& to, CellTO const& cellTO, Cell* cell);
    __inline__ __device__ void changeParticleFromTO(ParticleTO const& particleTO, Particle* particle);

    __inline__ __device__ Particle* createParticle(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Cell* createFreeCell(float energy, float2 const& pos, float2 const& vel);

    __inline__ __device__ Creature* cloneCreature(Creature* creature);

    __inline__ __device__ Cell*
    createCellFromNode(uint64_t& cellIndex, Creature* creature, int geneIndex, int nodeIndex, int parentNodeIndex, float2 pos, float2 vel, float energy);
    __inline__ __device__ Creature* createEmptyCreature();
    __inline__ __device__ Gene* createEmptyGenes(int numGenes);
    __inline__ __device__ Node* createEmptyNodes(int numNodes);

private:
    template<typename T>
    __inline__ __device__ void copyDataToHeap(T sourceSize, uint64_t sourceIndex, uint8_t* heap, T& targetSize, uint8_t*& target);
    __inline__ __device__ void copyDataToHeap(uint64_t size, uint64_t sourceIndex, uint8_t* source, uint8_t*& target);

    BaseMap _map;
    SimulationData* _data;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ObjectFactory::init(SimulationData* data)
{
    _data = data;
    _map.init(data->worldSize);
}

__inline__ __device__ Particle* ObjectFactory::createParticleFromTO(ParticleTO const& particleTO)
{
    Particle** particlePointer = _data->objects.particles.getNewElement();
    Particle* particle = _data->objects.heap.getTypedSubArray<Particle>(1);
    *particlePointer = particle;
    
    particle->id = particleTO.id;
    particle->pos = particleTO.pos;
    _map.correctPosition(particle->pos);
    particle->vel = particleTO.vel;
    particle->energy = particleTO.energy;
    particle->locked = 0;
    particle->selected = 0;
    particle->color = particleTO.color;
    particle->lastAbsorbedCell = nullptr;
    return particle;
}

__inline__ __device__ Genome* ObjectFactory::createGenomeFromTO(TO const& to, int genomeIndex)
{
    auto& genomeTO = to.genomes[genomeIndex];
    auto genome = _data->objects.heap.getTypedSubArray<Genome>(1);
    genomeTO.genomeIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(genome) - _data->objects.heap.getArray());
    genome->id = genomeTO.id;
    genome->frontAngle = genomeTO.frontAngle;
    genome->numGenes = genomeTO.numGenes;
    for (int i = 0; i < sizeof(genomeTO.name); ++i) {
        genome->name[i] = genomeTO.name[i];
    }

    auto const& geneTOs = to.genes + genomeTO.geneArrayIndex;
    auto genes = _data->objects.heap.getTypedSubArray<Gene>(genomeTO.numGenes);
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
        auto nodes = _data->objects.heap.getTypedSubArray<Node>(geneTO.numNodes);
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
            node.signalRestriction.active = nodeTO.signalRestriction.active;
            node.signalRestriction.baseAngle = nodeTO.signalRestriction.baseAngle;
            node.signalRestriction.openingAngle = nodeTO.signalRestriction.openingAngle;

            node.cellType = nodeTO.cellType;

            switch (nodeTO.cellType) {
            case CellTypeGenome_Base:
                break;
            case CellTypeGenome_Depot:
                node.cellTypeData.depot.mode = nodeTO.cellTypeData.depot.mode;
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
                node.cellTypeData.sensor.minDensity = nodeTO.cellTypeData.sensor.minDensity;
                node.cellTypeData.sensor.minRange = nodeTO.cellTypeData.sensor.minRange;
                node.cellTypeData.sensor.maxRange = nodeTO.cellTypeData.sensor.maxRange;
                node.cellTypeData.sensor.restrictToColor = nodeTO.cellTypeData.sensor.restrictToColor;
                node.cellTypeData.sensor.restrictToCreatures = nodeTO.cellTypeData.sensor.restrictToCreatures;
                break;
            case CellTypeGenome_Generator:
                node.cellTypeData.generator.autoTriggerInterval = nodeTO.cellTypeData.generator.autoTriggerInterval;
                node.cellTypeData.generator.pulseType = nodeTO.cellTypeData.generator.pulseType;
                node.cellTypeData.generator.alternationInterval = nodeTO.cellTypeData.generator.alternationInterval;
                break;
            case CellTypeGenome_Attacker:
                break;
            case CellTypeGenome_Injector:
                node.cellTypeData.injector.mode = nodeTO.cellTypeData.injector.mode;
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
                    node.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio = nodeTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                    break;
                case MuscleMode_AngleBending:
                    node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = nodeTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                    node.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio = nodeTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                    break;
                case MuscleMode_AutoCrawling:
                    node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = nodeTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                    node.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = nodeTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                    break;
                case MuscleMode_ManualCrawling:
                    node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                        nodeTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                    node.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio = nodeTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                    break;
                case MuscleMode_DirectMovement:
                    break;
                }
                break;
            case CellTypeGenome_Defender:
                node.cellTypeData.defender.mode = nodeTO.cellTypeData.defender.mode;
                break;
            case CellTypeGenome_Reconnector:
                node.cellTypeData.reconnector.restrictToColor = nodeTO.cellTypeData.reconnector.restrictToColor;
                node.cellTypeData.reconnector.restrictToCreatures = nodeTO.cellTypeData.reconnector.restrictToCreatures;
                break;
            case CellTypeGenome_Detonator:
                node.cellTypeData.detonator.countdown = nodeTO.cellTypeData.detonator.countdown;
                break;
            }
        }
    }
    return genome;
}

__inline__ __device__ Creature* ObjectFactory::createCreatureFromTO(TO const& to, int creatureIndex)
{
    auto& creatureTO = to.creatures[creatureIndex];
    auto creature = _data->objects.heap.getTypedSubArray<Creature>(1);
    creatureTO.creatureIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(creature) - _data->objects.heap.getArray());

    creature->id = creatureTO.id;
    creature->ancestorId = creatureTO.ancestorId;
    creature->generation = creatureTO.generation;
    creature->lineageId = creatureTO.lineageId;
    creature->numCells = creatureTO.numCells;
    creature->frontAngleId = creatureTO.frontAngleId;

    auto const& genomeTO = to.genomes[creatureTO.genomeArrayIndex];
    creature->genome = &_data->objects.heap.atType<Genome>(genomeTO.genomeIndexOnGpu);

    return creature;
}

__inline__ __device__ Cell* ObjectFactory::createCellFromTO(TO const& to, int cellIndex, Cell* cellArray)
{
    auto cellTO = to.cells[cellIndex];
    Cell** cellPointer = _data->objects.cells.getNewElement();
    Cell* cell = cellArray + cellIndex;
    *cellPointer = cell;

    changeCellFromTO(to, cellTO, cell);
    cell->id = cellTO.id;
    cell->locked = 0;
    cell->detached = 0;
    cell->selected = 0;
    cell->scheduledOperationIndex = -1;
    cell->numConnections = cellTO.numConnections;
    cell->event = CellEvent_No;
    cell->density = 1.0f;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectingCell = cell->connections[i];
        connectingCell.cell = cellArray + cellTO.connections[i].cellIndex;
        connectingCell.distance = cellTO.connections[i].distance;
        connectingCell.angleFromPrevious = cellTO.connections[i].angleFromPrevious;
    }
    if (cellTO.belongToCreature) {
        auto const& genomeTO = to.creatures[cellTO.creatureIndex];
        cell->creature = &_data->objects.heap.atType<Creature>(genomeTO.creatureIndexOnGpu);
    } else {
        cell->creature = nullptr;
    }
    return cell;
}

__inline__ __device__ void ObjectFactory::changeCellFromTO(TO const& to, CellTO const& cellTO, Cell* cell)
{
    cell->id = cellTO.id;
    cell->pos = cellTO.pos;
    _map.correctPosition(cell->pos);
    cell->vel = cellTO.vel;
    cell->cellState = cellTO.cellState;
    cell->energy = cellTO.energy;
    cell->stiffness = cellTO.stiffness;
    cell->cellType = cellTO.cellType;
    cell->barrier = cellTO.barrier;
    cell->sticky = cellTO.sticky;
    cell->age = cellTO.age;
    cell->color = cellTO.color;
    cell->frontAngle = cellTO.frontAngle;
    cell->activationTime = cellTO.activationTime;
    cell->detectedByCreatureId = cellTO.detectedByCreatureId;
    cell->cellTriggered = cellTO.cellTriggered;
    cell->nodeIndex = cellTO.nodeIndex;
    cell->parentNodeIndex = cellTO.parentNodeIndex;
    cell->geneIndex = cellTO.geneIndex;
    cell->frontAngleId = cellTO.frontAngleId;
    cell->isFrontAngleRefCell = cellTO.isFrontAngleRefCell;

    cell->signalRestriction.active = cellTO.signalRestriction.active;
    cell->signalRestriction.baseAngle = cellTO.signalRestriction.baseAngle;
    cell->signalRestriction.openingAngle = cellTO.signalRestriction.openingAngle;

    cell->signalState = cellTO.signalState;
    cell->signal.active = cellTO.signal.active;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->signal.channels[i] = cellTO.signal.channels[i];
    }

    cell->cellType = cellTO.cellType;

    if (cellTO.neuralNetworkDataIndex != VALUE_NOT_SET_UINT64) {
        copyDataToHeap(
            sizeof(NeuralNetworkTO), cellTO.neuralNetworkDataIndex, to.heap, reinterpret_cast<uint8_t*&>(cell->neuralNetwork));
    } else {
        cell->neuralNetwork = nullptr;
    }
    switch (cellTO.cellType) {
    case CellType_Base: {
    } break;
    case CellType_Depot: {
        cell->cellTypeData.depot.mode = cellTO.cellTypeData.depot.mode;
    } break;
    case CellType_Constructor: {
        cell->cellTypeData.constructor.autoTriggerInterval = cellTO.cellTypeData.constructor.autoTriggerInterval;
        cell->cellTypeData.constructor.constructionActivationTime = cellTO.cellTypeData.constructor.constructionActivationTime;
        cell->cellTypeData.constructor.constructionAngle = cellTO.cellTypeData.constructor.constructionAngle;
        cell->cellTypeData.constructor.provideEnergy = cellTO.cellTypeData.constructor.provideEnergy;
        cell->cellTypeData.constructor.geneIndex = cellTO.cellTypeData.constructor.geneIndex;
        cell->cellTypeData.constructor.lastConstructedCellId = cellTO.cellTypeData.constructor.lastConstructedCellId;
        cell->cellTypeData.constructor.currentNodeIndex = cellTO.cellTypeData.constructor.currentNodeIndex;
        cell->cellTypeData.constructor.currentConcatenation = cellTO.cellTypeData.constructor.currentConcatenation;
        cell->cellTypeData.constructor.currentBranch = cellTO.cellTypeData.constructor.currentBranch;
        cell->cellTypeData.constructor.isReady = true;
        cell->cellTypeData.constructor.offspring = nullptr;
    } break;
    case CellType_Sensor: {
        cell->cellTypeData.sensor.autoTriggerInterval = cellTO.cellTypeData.sensor.autoTriggerInterval;
        cell->cellTypeData.sensor.minDensity = cellTO.cellTypeData.sensor.minDensity;
        cell->cellTypeData.sensor.minRange = cellTO.cellTypeData.sensor.minRange;
        cell->cellTypeData.sensor.maxRange = cellTO.cellTypeData.sensor.maxRange;
        cell->cellTypeData.sensor.restrictToColor = cellTO.cellTypeData.sensor.restrictToColor;
        cell->cellTypeData.sensor.restrictToCreatures = cellTO.cellTypeData.sensor.restrictToCreatures;
    } break;
    case CellType_Generator: {
        cell->cellTypeData.generator.autoTriggerInterval = cellTO.cellTypeData.generator.autoTriggerInterval;
        cell->cellTypeData.generator.pulseType = cellTO.cellTypeData.generator.pulseType;
        cell->cellTypeData.generator.alternationInterval = cellTO.cellTypeData.generator.alternationInterval;
        cell->cellTypeData.generator.numPulses = cellTO.cellTypeData.generator.numPulses;
    } break;
    case CellType_Attacker: {
    } break;
    case CellType_Injector: {
        cell->cellTypeData.injector.mode = cellTO.cellTypeData.injector.mode;
        cell->cellTypeData.injector.counter = cellTO.cellTypeData.injector.counter;
    } break;
    case CellType_Muscle: {
        cell->cellTypeData.muscle.mode = cellTO.cellTypeData.muscle.mode;
        if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
            cell->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            cell->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
            cell->cellTypeData.muscle.modeData.autoBending.initialAngle = cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle;
            cell->cellTypeData.muscle.modeData.autoBending.forward = cellTO.cellTypeData.muscle.modeData.autoBending.forward;
            cell->cellTypeData.muscle.modeData.autoBending.activation = cellTO.cellTypeData.muscle.modeData.autoBending.activation;
            cell->cellTypeData.muscle.modeData.autoBending.activationCountdown = cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown;
            cell->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            cell->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            cell->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
            cell->cellTypeData.muscle.modeData.manualBending.initialAngle = cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle;
            cell->cellTypeData.muscle.modeData.manualBending.lastAngleDelta = cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
            cell->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            cell->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            cell->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio = cellTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
            cell->cellTypeData.muscle.modeData.angleBending.initialAngle = cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
            cell->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            cell->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
            cell->cellTypeData.muscle.modeData.autoCrawling.initialDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
            cell->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
            cell->cellTypeData.muscle.modeData.autoCrawling.forward = cellTO.cellTypeData.muscle.modeData.autoCrawling.forward;
            cell->cellTypeData.muscle.modeData.autoCrawling.activation = cellTO.cellTypeData.muscle.modeData.autoCrawling.activation;
            cell->cellTypeData.muscle.modeData.autoCrawling.activationCountdown = cellTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
            cell->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
            cell->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            cell->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
            cell->cellTypeData.muscle.modeData.manualCrawling.initialDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance;
            cell->cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
            cell->cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = cellTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
            cell->cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
        }
        cell->cellTypeData.muscle.lastMovementX = cellTO.cellTypeData.muscle.lastMovementX;
        cell->cellTypeData.muscle.lastMovementY = cellTO.cellTypeData.muscle.lastMovementY;
    } break;
    case CellType_Defender: {
        cell->cellTypeData.defender.mode = cellTO.cellTypeData.defender.mode;
    } break;
    case CellType_Reconnector: {
        cell->cellTypeData.reconnector.restrictToColor = cellTO.cellTypeData.reconnector.restrictToColor;
        cell->cellTypeData.reconnector.restrictToCreatures = cellTO.cellTypeData.reconnector.restrictToCreatures;
    } break;
    case CellType_Detonator: {
        cell->cellTypeData.detonator.state = cellTO.cellTypeData.detonator.state;
        cell->cellTypeData.detonator.countdown = cellTO.cellTypeData.detonator.countdown;
    } break;
    }
}

__inline__ __device__ void ObjectFactory::changeParticleFromTO(ParticleTO const& particleTO, Particle* particle)
{
    particle->energy = particleTO.energy;
    particle->pos = particleTO.pos;
    particle->color = particleTO.color;
}

template <typename T>
__inline__ __device__ void ObjectFactory::copyDataToHeap(T sourceSize, uint64_t sourceIndex, uint8_t* heap, T& targetSize, uint8_t*& target)
{
    targetSize = sourceSize;
    copyDataToHeap(sourceSize, sourceIndex, heap, target);
}

__inline__ __device__ void ObjectFactory::copyDataToHeap(uint64_t size, uint64_t sourceIndex, uint8_t* source, uint8_t*& target)
{
    if (size > 0) {
        target = _data->objects.heap.getRawSubArray(size);
        for (int i = 0; i < size; ++i) {
            target[i] = source[sourceIndex + i];
        }
    }
}

__inline__ __device__ Particle*
ObjectFactory::createParticle(float energy, float2 const& pos, float2 const& vel, int color)
{
    Particle** particlePointer = _data->objects.particles.getNewElement();
    Particle* particle = _data->objects.heap.getTypedSubArray<Particle>(1);
    *particlePointer = particle;
    particle->id = _data->primaryNumberGen.createObjectId();
    particle->selected = 0;
    particle->locked = 0;
    particle->energy = energy;
    particle->pos = pos;
    particle->vel = vel;
    particle->color = color;
    particle->lastAbsorbedCell = nullptr;
    return particle;
}

__inline__ __device__ Cell* ObjectFactory::createFreeCell(float energy, float2 const& pos, float2 const& vel)
{
    auto cell = _data->objects.heap.getTypedSubArray<Cell>(1);
    auto cellPointers = _data->objects.cells.getNewElement();
    *cellPointers = cell;

    cell->id = _data->primaryNumberGen.createObjectId();
    cell->pos = pos;
    cell->vel = vel;
    cell->energy = energy;
    cell->stiffness = _data->primaryNumberGen.random();
    cell->numConnections = 0;
    cell->cellState = CellState_Ready;
    cell->locked = 0;
    cell->selected = 0;
    cell->detached = 0;
    cell->scheduledOperationIndex = -1;
    cell->color = 0;
    cell->frontAngle = VALUE_NOT_SET_FLOAT;
    cell->frontAngleId = 0;
    cell->isFrontAngleRefCell = false;
    cell->barrier = false;
    cell->sticky = false;
    cell->age = 0;
    cell->activationTime = 0;
    cell->signalRestriction.active = false;
    cell->signalState = 0;
    cell->signal.active = false;
    cell->density = 1.0f;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->cellTriggered = CellTriggered_No;
    cell->creature = nullptr;
    cell->nodeIndex = 0;
    cell->parentNodeIndex = 0;
    cell->geneIndex = 0;
    cell->cellType = CellType_Free;
    cell->neuralNetwork = nullptr;
    
    return cell;
}

__inline__ __device__ Creature* ObjectFactory::cloneCreature(Creature* creature)
{
    auto newCreature = createEmptyCreature();
    auto newId = newCreature->id;
    *newCreature = *creature;
    newCreature->id = newId;
    newCreature->ancestorId = creature->id;
    newCreature->generation = creature->generation + 1;
    return newCreature;
}

//__inline__ __device__ Cell* ObjectFactory::createEmptyCell(uint64_t& cellIndex)
//{
//    auto cell = _data->objects.heap.getTypedSubArray<Cell>(1);
//    auto cellPointer = _data->objects.cells.getNewElement(&cellIndex);
//    *cellPointer = cell;
//
//    cell->id = _data->primaryNumberGen.createObjectId();
//    cell->stiffness = 1.0f;
//    cell->selected = 0;
//    cell->detached = 0;
//    cell->scheduledOperationIndex = -1;
//    cell->locked = 0;
//    cell->color = 0;
//    cell->barrier = false;
//    cell->sticky = false;
//    cell->age = 0;
//    cell->vel = {0, 0};
//    cell->activationTime = 0;
//    cell->signalRestriction.active = false;
//    cell->signalState = 0;
//    cell->signal.active = false;
//    cell->density = 1.0f;
//    cell->detectedByCreatureId = 0;
//    cell->event = CellEvent_No;
//    cell->cellTriggered = CellTriggered_No;
//    return cell;
//}

__inline__ __device__ Cell*
ObjectFactory::createCellFromNode(uint64_t& cellIndex, Creature* creature, int geneIndex, int nodeIndex, int parentNodeIndex, float2 pos, float2 vel, float energy)
{
    auto const& gene = &creature->genome->genes[geneIndex];
    auto const& node = &gene->nodes[nodeIndex];

    auto cell = _data->objects.heap.getTypedSubArray<Cell>(1);
    auto cellPointer = _data->objects.cells.getNewElement(&cellIndex);
    *cellPointer = cell;
    cell->id = _data->primaryNumberGen.createObjectId();
    cell->pos = pos;
    cell->vel = vel;
    cell->energy = energy;
    cell->stiffness = gene->stiffness;
    cell->color = node->color;
    cell->frontAngle = VALUE_NOT_SET_FLOAT;
    cell->stiffness = 1.0f;
    cell->barrier = false;
    cell->sticky = false;
    cell->age = 0;
    cell->cellState = CellState_Constructing;
    cell->creature = creature;
    cell->nodeIndex = nodeIndex;
    cell->parentNodeIndex = parentNodeIndex;
    cell->geneIndex = geneIndex;
    cell->numConnections = 0;
    cell->frontAngleId = 0;
    cell->isFrontAngleRefCell = false;

    cell->neuralNetwork = _data->objects.heap.getTypedSubArray<NeuralNetwork>(1);
    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        cell->neuralNetwork->weights[i] = node->neuralNetwork.weights[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->neuralNetwork->biases[i] = node->neuralNetwork.biases[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->neuralNetwork->activationFunctions[i] = node->neuralNetwork.activationFunctions[i];
    }
    cell->signalRestriction.active = node->signalRestriction.active;
    cell->signalRestriction.baseAngle = node->signalRestriction.baseAngle;
    cell->signalRestriction.openingAngle = node->signalRestriction.openingAngle;
    cell->signalState = 0;
    cell->signal.active = false;
    cell->activationTime = 0;
    cell->cellTriggered = CellTriggered_No;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->selected = 0;
    cell->detached = 0;
    cell->locked = 0;
    cell->density = 1.0f;
    cell->scheduledOperationIndex = -1;

    switch (node->cellType) {
    case CellTypeGenome_Base: {
        cell->cellType = CellType_Base;
    } break;
    case CellTypeGenome_Depot: {
        cell->cellType = CellType_Depot;
        cell->cellTypeData.depot.mode = node->cellTypeData.depot.mode;
    } break;
    case CellTypeGenome_Constructor: {
        cell->cellType = CellType_Constructor;
        auto const& nodeConstructor = node->cellTypeData.constructor;
        auto& constructor = cell->cellTypeData.constructor;
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
        cell->cellType = CellType_Sensor;
        auto const& nodeSensor = node->cellTypeData.sensor;
        auto& sensor = cell->cellTypeData.sensor;
        sensor.autoTriggerInterval = nodeSensor.autoTriggerInterval;
        sensor.minDensity = nodeSensor.minDensity;
        sensor.minRange = nodeSensor.minRange;
        sensor.maxRange = nodeSensor.maxRange;
        sensor.restrictToColor = nodeSensor.restrictToColor;
        sensor.restrictToCreatures = nodeSensor.restrictToCreatures;
    } break;
    case CellTypeGenome_Generator: {
        cell->cellType = CellType_Generator;
        auto const& nodeGenerator = node->cellTypeData.generator;
        auto& generator = cell->cellTypeData.generator;
        generator.autoTriggerInterval = nodeGenerator.autoTriggerInterval;
        generator.pulseType = nodeGenerator.pulseType;
        generator.alternationInterval = nodeGenerator.alternationInterval;
        generator.numPulses = 0;
    } break;
    case CellTypeGenome_Attacker: {
        cell->cellType = CellType_Attacker;
    } break;
    case CellTypeGenome_Injector: {
        cell->cellType = CellType_Injector;
        auto const& nodeInjector = node->cellTypeData.injector;
        auto& injector = cell->cellTypeData.injector;
        injector.mode = nodeInjector.mode;
        injector.counter = 0;
    } break;
    case CellTypeGenome_Muscle: {
        cell->cellType = CellType_Muscle;
        auto const& nodeMuscle = node->cellTypeData.muscle;
        auto& muscle = cell->cellTypeData.muscle;
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
        cell->cellType = CellType_Defender;
        auto const& nodeDefender = node->cellTypeData.defender;
        auto& defender = cell->cellTypeData.defender;
        defender.mode = nodeDefender.mode;
    } break;
    case CellTypeGenome_Reconnector: {
        cell->cellType = CellType_Reconnector;
        auto const& nodeReconnector = node->cellTypeData.reconnector;
        auto& reconnector = cell->cellTypeData.reconnector;
        reconnector.restrictToColor = nodeReconnector.restrictToColor;
        reconnector.restrictToCreatures = nodeReconnector.restrictToCreatures;
    } break;
    case CellTypeGenome_Detonator: {
        cell->cellType = CellType_Detonator;
        auto const& nodeDetonator = node->cellTypeData.detonator;
        auto& detonator = cell->cellTypeData.detonator;
        detonator.state = DetonatorState_Ready;
        detonator.countdown = nodeDetonator.countdown;
    } break;
    }
    return cell;
}

__inline__ __device__ Creature* ObjectFactory::createEmptyCreature()
{
    auto creature = _data->objects.heap.getTypedSubArray<Creature>(1);
    creature->id = _data->primaryNumberGen.createCreatureId();
    return creature;
}

__inline__ __device__ Gene* ObjectFactory::createEmptyGenes(int numGenes)
{
    auto genes = _data->objects.heap.getTypedSubArray<Gene>(numGenes);
    return genes;
}

__inline__ __device__ Node* ObjectFactory::createEmptyNodes(int numNodes)
{
    auto nodes = _data->objects.heap.getTypedSubArray<Node>(numNodes);
    return nodes;
}
