#pragma once

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "ObjectTO.cuh"
#include "Map.cuh"
#include "Object.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"

class ObjectFactory
{
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Particle* createParticleFromTO(ParticleTO const& particleTO);
    __inline__ __device__ Creature* createCreatureFromTO(CollectionTO const& collectionTO, int creatureIndex);
    __inline__ __device__ Cell* createCellFromTO(CollectionTO const& collectionTO, int cellIndex, Cell* cellArray);
    __inline__ __device__ void changeCellFromTO(CollectionTO const& collectionTO, CellTO const& cellTO, Cell* cell);
    __inline__ __device__ void changeParticleFromTO(ParticleTO const& particleTO, Particle* particle);

    __inline__ __device__ Particle* createParticle(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Cell* createFreeCell(float energy, float2 const& pos, float2 const& vel);

    __inline__ __device__ Creature* cloneCreature(Creature* creature);

    __inline__ __device__ Cell* createEmptyCell(uint64_t& cellIndex);
    __inline__ __device__ Cell* createCellFromNode(uint64_t& cellIndex, Creature* creature, int geneIndex, int nodeIndex, float2 pos, float2 vel, float energy);
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

__inline__ __device__ Creature* ObjectFactory::createCreatureFromTO(CollectionTO const& collectionTO, int creatureIndex)
{
    auto& creatureTO = collectionTO.creatures[creatureIndex];
    auto creature = _data->objects.heap.getTypedSubArray<Creature>(1);
    creatureTO.creatureIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(creature) - _data->objects.heap.getArray());

    creature->id = creatureTO.id;
    creature->ancestorId = creatureTO.ancestorId;
    creature->mutationId = creatureTO.mutationId;
    creature->genomeComplexity = creatureTO.genomeComplexity;
    creature->genome.frontAngle = creatureTO.genome.frontAngle;
    creature->genome.numGenes = creatureTO.genome.numGenes;

    auto const& geneTOs = collectionTO.genes + creatureTO.genome.geneArrayIndex;
    auto genes = _data->objects.heap.getTypedSubArray<Gene>(creatureTO.genome.numGenes);
    creature->genome.genes = genes;
    for (int i = 0, j = creatureTO.genome.numGenes; i < j; ++i) {
        auto const& geneTO = geneTOs[i];
        auto& gene = genes[i];
        gene.shape = geneTO.shape;
        gene.numBranches = geneTO.numBranches;
        gene.angleAlignment = geneTO.angleAlignment;
        gene.stiffness = geneTO.stiffness;
        gene.connectionDistance = geneTO.connectionDistance;
        gene.numConcatenations = geneTO.numConcatenations;
        gene.numNodes = geneTO.numNodes;

        auto const& nodeTOs = collectionTO.nodes + geneTO.nodeArrayIndex;
        auto nodes = _data->objects.heap.getTypedSubArray<Node>(geneTO.numNodes);
        gene.nodes = nodes;
        for (int i = 0, j = geneTO.numNodes; i < j; ++i) {
            auto const& nodeTO = nodeTOs[i];
            auto& node = nodes[i];
            node.referenceAngle = nodeTO.referenceAngle;
            node.color = nodeTO.color;
            node.numRequiredAdditionalConnections = nodeTO.numRequiredAdditionalConnections;

            copyDataToHeap(sizeof(NeuralNetworkGenomeTO), nodeTO.neuralNetworkDataIndex, collectionTO.heap, reinterpret_cast<uint8_t*&>(node.neuralNetwork));
            node.signalRoutingRestriction.active = nodeTO.signalRoutingRestriction.active;
            node.signalRoutingRestriction.baseAngle = nodeTO.signalRoutingRestriction.baseAngle;
            node.signalRoutingRestriction.openingAngle = nodeTO.signalRoutingRestriction.openingAngle;

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
                    node.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio = nodeTO.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
                    break;
                case MuscleMode_ManualBending:
                    node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = nodeTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                    node.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio = nodeTO.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio;
                    break;
                case MuscleMode_AngleBending:
                    node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = nodeTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                    node.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio = nodeTO.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio;
                    break;
                case MuscleMode_AutoCrawling:
                    node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = nodeTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                    node.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio = nodeTO.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio;
                    break;
                case MuscleMode_ManualCrawling:
                    node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                        nodeTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                    node.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio = nodeTO.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio;
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
    return creature;
}

__inline__ __device__ Cell* ObjectFactory::createCellFromTO(CollectionTO const& collectionTO, int cellIndex, Cell* cellArray)
{
    auto cellTO = collectionTO.cells[cellIndex];
    Cell** cellPointer = _data->objects.cells.getNewElement();
    Cell* cell = cellArray + cellIndex;
    *cellPointer = cell;

    changeCellFromTO(collectionTO, cellTO, cell);
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
        auto const& genomeTO = collectionTO.creatures[cellTO.creatureIndex];
        cell->creature = &_data->objects.heap.atType<Creature>(genomeTO.creatureIndexOnGpu);
    } else {
        cell->creature = nullptr;
    }
    return cell;
}

__inline__ __device__ void ObjectFactory::changeCellFromTO(CollectionTO const& collectionTO, CellTO const& cellTO, Cell* cell)
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
    cell->angleToFront = cellTO.angleToFront;
    cell->activationTime = cellTO.activationTime;
    cell->detectedByCreatureId = cellTO.detectedByCreatureId;
    cell->cellTriggered = cellTO.cellTriggered;
    cell->genomeNodeIndex = cellTO.genomeNodeIndex;

    copyDataToHeap(cellTO.metadata.nameSize, cellTO.metadata.nameDataIndex, collectionTO.heap, cell->metadata.nameSize, cell->metadata.name);

    copyDataToHeap(
        cellTO.metadata.descriptionSize,
        cellTO.metadata.descriptionDataIndex,
        collectionTO.heap,
        cell->metadata.descriptionSize,
        cell->metadata.description);

    cell->signalRoutingRestriction.active = cellTO.signalRoutingRestriction.active;
    cell->signalRoutingRestriction.baseAngle = cellTO.signalRoutingRestriction.baseAngle;
    cell->signalRoutingRestriction.openingAngle = cellTO.signalRoutingRestriction.openingAngle;

    cell->signalRelaxationTime = cellTO.signalRelaxationTime;
    cell->signal.active = cellTO.signal.active;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->signal.channels[i] = cellTO.signal.channels[i];
    }

    cell->cellType = cellTO.cellType;

    if (cellTO.neuralNetworkDataIndex != CellTO::NeuralNetworkDataIndex_NotSet) {
        copyDataToHeap(
            sizeof(NeuralNetworkTO), cellTO.neuralNetworkDataIndex, collectionTO.heap, reinterpret_cast<uint8_t*&>(cell->neuralNetwork));
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
        cell->cellTypeData.constructor.geneIndex = cellTO.cellTypeData.constructor.geneIndex;
        cell->cellTypeData.constructor.lastConstructedCellId = cellTO.cellTypeData.constructor.lastConstructedCellId;
        cell->cellTypeData.constructor.currentNodeIndex = cellTO.cellTypeData.constructor.currentNodeIndex;
        cell->cellTypeData.constructor.currentConcatenation = cellTO.cellTypeData.constructor.currentConcatenation;
        cell->cellTypeData.constructor.currentBranch = cellTO.cellTypeData.constructor.currentBranch;
        cell->cellTypeData.constructor.generation = cellTO.cellTypeData.constructor.generation;
        cell->cellTypeData.constructor.constructionAngle = cellTO.cellTypeData.constructor.constructionAngle;
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
            cell->cellTypeData.muscle.modeData.autoBending.frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
            cell->cellTypeData.muscle.modeData.autoBending.initialAngle = cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle;
            cell->cellTypeData.muscle.modeData.autoBending.lastActualAngle = cellTO.cellTypeData.muscle.modeData.autoBending.lastActualAngle;
            cell->cellTypeData.muscle.modeData.autoBending.forward = cellTO.cellTypeData.muscle.modeData.autoBending.forward;
            cell->cellTypeData.muscle.modeData.autoBending.activation = cellTO.cellTypeData.muscle.modeData.autoBending.activation;
            cell->cellTypeData.muscle.modeData.autoBending.activationCountdown = cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown;
            cell->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            cell->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            cell->cellTypeData.muscle.modeData.manualBending.frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio;
            cell->cellTypeData.muscle.modeData.manualBending.initialAngle = cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle;
            cell->cellTypeData.muscle.modeData.manualBending.lastActualAngle = cellTO.cellTypeData.muscle.modeData.manualBending.lastActualAngle;
            cell->cellTypeData.muscle.modeData.manualBending.lastAngleDelta = cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
            cell->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            cell->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            cell->cellTypeData.muscle.modeData.angleBending.frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio;
            cell->cellTypeData.muscle.modeData.angleBending.initialAngle = cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
            cell->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            cell->cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio;
            cell->cellTypeData.muscle.modeData.autoCrawling.initialDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
            cell->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
            cell->cellTypeData.muscle.modeData.autoCrawling.forward = cellTO.cellTypeData.muscle.modeData.autoCrawling.forward;
            cell->cellTypeData.muscle.modeData.autoCrawling.activation = cellTO.cellTypeData.muscle.modeData.autoCrawling.activation;
            cell->cellTypeData.muscle.modeData.autoCrawling.activationCountdown = cellTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
            cell->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
            cell->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            cell->cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio;
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
    cell->angleToFront = 0;
    cell->metadata.nameSize = 0;
    cell->metadata.descriptionSize = 0;
    cell->barrier = false;
    cell->sticky = false;
    cell->age = 0;
    cell->activationTime = 0;
    cell->signalRoutingRestriction.active = false;
    cell->signalRelaxationTime = 0;
    cell->signal.active = false;
    cell->density = 1.0f;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->cellTriggered = CellTriggered_No;
    cell->genomeNodeIndex = 0;
    cell->cellType = CellType_Free;
    cell->neuralNetwork = nullptr;

    return cell;
}

__inline__ __device__ Creature* ObjectFactory::cloneCreature(Creature* creature)
{
    auto newCreature = createEmptyCreature();
    *newCreature = *creature;
    newCreature->ancestorId = creature->id;
    auto genes = createEmptyGenes(creature->genome.numGenes);
    newCreature->genome.genes = genes;

    for (int i = 0, numGenes = creature->genome.numGenes; i < numGenes; ++i) {
        auto gene = &creature->genome.genes[i];
        auto newGene = &newCreature->genome.genes[i];
        *newGene = *gene;
        auto nodes = createEmptyNodes(gene->numNodes);
        newGene->numNodes = gene->numNodes;
        newGene->nodes = nodes;
        for (int j = 0, numNodes = gene->numNodes; j < numNodes; ++j) {
            auto node = &gene->nodes[i];
            auto newNode = &newGene->nodes[i];
            *newNode = *node;
        }
    }

    return newCreature;
}

__inline__ __device__ Cell* ObjectFactory::createEmptyCell(uint64_t& cellIndex)
{
    auto cell = _data->objects.heap.getTypedSubArray<Cell>(1);
    auto cellPointer = _data->objects.cells.getNewElement(&cellIndex);
    *cellPointer = cell;

    cell->id = _data->primaryNumberGen.createObjectId();
    cell->stiffness = 1.0f;
    cell->selected = 0;
    cell->detached = 0;
    cell->scheduledOperationIndex = -1;
    cell->locked = 0;
    cell->color = 0;
    cell->metadata.nameSize = 0;
    cell->metadata.descriptionSize = 0;
    cell->barrier = false;
    cell->sticky = false;
    cell->age = 0;
    cell->vel = {0, 0};
    cell->activationTime = 0;
    cell->signalRoutingRestriction.active = false;
    cell->signalRelaxationTime = 0;
    cell->signal.active = false;
    cell->density = 1.0f;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->cellTriggered = CellTriggered_No;
    return cell;
}

__inline__ __device__ Cell* ObjectFactory::createCellFromNode(uint64_t& cellIndex, Creature* creature, int geneIndex, int nodeIndex, float2 pos, float2 vel, float energy)
{
    auto const& gene = &creature->genome.genes[geneIndex];
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
    cell->angleToFront = 0;
    cell->stiffness = 1.0f;
    cell->barrier = false;
    cell->sticky = false;
    cell->age = 0;
    cell->cellState = CellState_Constructing;
    cell->creature = creature;
    cell->genomeNodeIndex = nodeIndex;

    cell->neuralNetwork = _data->objects.heap.getTypedSubArray<NeuralNetwork>(1);
    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        cell->neuralNetwork->weights[i] = node->neuralNetwork->weights[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->neuralNetwork->biases[i] = node->neuralNetwork->biases[i];
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->neuralNetwork->activationFunctions[i] = node->neuralNetwork->activationFunctions[i];
    }
    cell->signalRoutingRestriction.active = node->signalRoutingRestriction.active;
    cell->signalRoutingRestriction.baseAngle = node->signalRoutingRestriction.baseAngle;
    cell->signalRoutingRestriction.openingAngle = node->signalRoutingRestriction.openingAngle;
    cell->signalRelaxationTime = 0;
    cell->signal.active = false;
    cell->activationTime = 0;
    cell->cellTriggered = CellTriggered_No;
    cell->detectedByCreatureId = 0;
    cell->metadata.nameSize = 0;
    cell->metadata.descriptionSize = 0;
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
        auto& newConstructor = cell->cellTypeData.constructor;
        newConstructor.autoTriggerInterval = nodeConstructor.autoTriggerInterval;
        newConstructor.constructionActivationTime = nodeConstructor.constructionActivationTime;
        newConstructor.geneIndex = nodeConstructor.geneIndex;
        newConstructor.constructionAngle = nodeConstructor.constructionAngle;
        newConstructor.lastConstructedCellId = 0;
        newConstructor.currentNodeIndex = 0;
        newConstructor.currentConcatenation = 0;
        newConstructor.currentBranch = 0;
        newConstructor.isReady = true;
        newConstructor.offspring = nullptr;
    } break;
    case CellTypeGenome_Sensor: {
        result->cellTypeData.sensor.autoTriggerInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.sensor.minDensity = (GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) + 1.0f) / 2;
        result->cellTypeData.sensor.restrictToColor = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition, MAX_COLORS);
        result->cellTypeData.sensor.restrictToCreatures = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % SensorRestrictToCreatures_Count;
        result->cellTypeData.sensor.minRange = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.sensor.maxRange = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition);
    } break;
    case CellTypeGenome_Generator: {
        result->cellTypeData.generator.autoTriggerInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.generator.alternationInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.generator.numPulses = 0;
    } break;
    case CellTypeGenome_Attacker: {
    } break;
    case CellTypeGenome_Injector: {
        result->cellTypeData.injector.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % InjectorMode_Count;
        result->cellTypeData.injector.counter = 0;
        GenomeDecoder::copyGenome(data, constructor, genomeCurrentBytePosition, result->cellTypeData.injector);
        result->cellTypeData.injector.generation = constructor.generation + 1;
    } break;
    case CellTypeGenome_Muscle: {
        result->cellTypeData.muscle.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % MuscleMode_Count;
        if (result->cellTypeData.muscle.mode == MuscleMode_AutoBending) {
            result->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation =
                min(1.0f, max(0.0f, GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition)));
            result->cellTypeData.muscle.modeData.autoBending.frontBackVelRatio = abs(GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.autoBending.initialAngle = 0;
            result->cellTypeData.muscle.modeData.autoBending.lastActualAngle = 0;
            result->cellTypeData.muscle.modeData.autoBending.forward = true;
            result->cellTypeData.muscle.modeData.autoBending.activation = 0;
            result->cellTypeData.muscle.modeData.autoBending.activationCountdown = 0;
            result->cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied = false;
        } else if (result->cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            result->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation =
                min(1.0f, max(0.0f, GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition)));
            result->cellTypeData.muscle.modeData.manualBending.frontBackVelRatio = abs(GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.manualBending.initialAngle = 0;
            result->cellTypeData.muscle.modeData.manualBending.lastActualAngle = 0;
            result->cellTypeData.muscle.modeData.manualBending.lastAngleDelta = 0;
            result->cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied = false;
        } else if (result->cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            result->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation =
                min(1.0f, max(0.0f, GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition)));
            result->cellTypeData.muscle.modeData.angleBending.frontBackVelRatio = abs(GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.angleBending.initialAngle = 0;
        } else if (result->cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
            result->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation =
                min(1.0f, max(0.0f, GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition)));
            result->cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio = abs(GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.autoCrawling.initialDistance = 0;
            result->cellTypeData.muscle.modeData.autoCrawling.lastActualDistance = 0;
            result->cellTypeData.muscle.modeData.autoCrawling.forward = true;
            result->cellTypeData.muscle.modeData.autoCrawling.activation = 0;
            result->cellTypeData.muscle.modeData.autoCrawling.activationCountdown = 0;
            result->cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied = false;
        } else if (result->cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
            result->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation =
                min(1.0f, max(0.0f, GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition)));
            result->cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio = abs(GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.manualCrawling.initialDistance = 0;
            result->cellTypeData.muscle.modeData.manualCrawling.lastActualDistance = 0;
            result->cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta = 0;
            result->cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied = false;
        } else if (result->cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
            // Read dummy bytes
            GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition);
            GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition);
        }
        result->cellTypeData.muscle.lastMovementX = 0;
        result->cellTypeData.muscle.lastMovementY = 0;
    } break;
    case CellTypeGenome_Defender: {
        result->cellTypeData.defender.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % DefenderMode_Count;
    } break;
    case CellTypeGenome_Reconnector: {
        result->cellTypeData.reconnector.restrictToColor = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition, MAX_COLORS);
        result->cellTypeData.reconnector.restrictToCreatures =
            GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % ReconnectorRestrictToCreatures_Count;
    } break;
    case CellTypeGenome_Detonator: {
        result->cellTypeData.detonator.state = DetonatorState_Ready;
        result->cellTypeData.detonator.countdown = GenomeDecoder::readWord(constructor, genomeCurrentBytePosition);
    } break;
    }
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
