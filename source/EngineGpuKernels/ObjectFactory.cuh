#pragma once

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/GenomeConstants.h"

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
    __inline__ __device__ Particle* createParticleFromTO(ParticleTO const& particleTO, bool createIds);
    __inline__ __device__ void createGenomeFromTO(CollectionTO const& collectionTO, int genomeIndex, bool createIds);
    __inline__ __device__ Cell* createCellFromTO(CollectionTO const& collectionTO, int cellIndex, Cell* cellArray, bool createIds);
    __inline__ __device__ void changeCellFromTO(CollectionTO const& collectionTO, CellTO const& cellTO, Cell* cell, bool createIds);
    __inline__ __device__ void changeParticleFromTO(ParticleTO const& particleTO, Particle* particle);
    __inline__ __device__ Particle* createParticle(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Cell* createFreeCell(float energy, float2 const& pos, float2 const& vel);
    __inline__ __device__ Cell* createCell(uint64_t& cellPointerIndex);

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

__inline__ __device__ Particle* ObjectFactory::createParticleFromTO(ParticleTO const& particleTO, bool createIds)
{
    Particle** particlePointer = _data->objects.particles.getNewElement();
    Particle* particle = _data->objects.heap.getTypedSubArray<Particle>(1);
    *particlePointer = particle;
    
    particle->id = createIds ? _data->numberGen1.createNewId() : particleTO.id;
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

__inline__ __device__ void ObjectFactory::createGenomeFromTO(CollectionTO const& collectionTO, int genomeIndex, bool createIds)
{
    auto& genomeTO = collectionTO.genomes[genomeIndex];
    auto genome = _data->objects.heap.getTypedSubArray<Genome>(1);
    genomeTO.genomeIndexOnGpu = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(genome) - _data->objects.heap.getArray());

    genome->frontAngle = genomeTO.frontAngle;
    genome->numGenes = genomeTO.numGenes;

    auto const& geneTOs = collectionTO.genes + genomeTO.geneArrayIndex;
    auto genes = _data->objects.heap.getTypedSubArray<Gene>(genomeTO.numGenes);
    genome->genes = genes;
    for (int i = 0, j = genomeTO.numGenes; i < j; ++i) {
        auto const& geneTO = geneTOs[i];
        auto& gene = genes[i];
        gene.shape = geneTO.shape;
        gene.numBranches = geneTO.numBranches;
        gene.separateConstruction = geneTO.separateConstruction;
        gene.angleAlignment = geneTO.angleAlignment;
        gene.stiffness = geneTO.stiffness;
        gene.connectionDistance = geneTO.connectionDistance;
        gene.numRepetitions = geneTO.numRepetitions;
        gene.concatenationAngle1 = geneTO.concatenationAngle1;
        gene.concatenationAngle2 = geneTO.concatenationAngle2;
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
                node.cellTypeData.constructor.constructionActivationTime = nodeTO.cellTypeData.constructor.constructionActivationTime;
                node.cellTypeData.constructor.constructionAngle1 = nodeTO.cellTypeData.constructor.constructionAngle1;
                node.cellTypeData.constructor.constructionAngle2 = nodeTO.cellTypeData.constructor.constructionAngle2;
                break;
            case CellTypeGenome_Sensor:
                node.cellTypeData.sensor.autoTriggerInterval = nodeTO.cellTypeData.sensor.autoTriggerInterval;
                node.cellTypeData.sensor.minDensity = nodeTO.cellTypeData.sensor.minDensity;
                node.cellTypeData.sensor.minRange = nodeTO.cellTypeData.sensor.minRange;
                node.cellTypeData.sensor.maxRange = nodeTO.cellTypeData.sensor.maxRange;
                node.cellTypeData.sensor.restrictToColor = nodeTO.cellTypeData.sensor.restrictToColor;
                node.cellTypeData.sensor.restrictToMutants = nodeTO.cellTypeData.sensor.restrictToMutants;
                break;
            case CellTypeGenome_Oscillator:
                node.cellTypeData.oscillator.autoTriggerInterval = nodeTO.cellTypeData.oscillator.autoTriggerInterval;
                node.cellTypeData.oscillator.alternationInterval = nodeTO.cellTypeData.oscillator.alternationInterval;
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
                node.cellTypeData.reconnector.restrictToMutants = nodeTO.cellTypeData.reconnector.restrictToMutants;
                break;
            case CellTypeGenome_Detonator:
                node.cellTypeData.detonator.countdown = nodeTO.cellTypeData.detonator.countdown;
                break;
            }
        }
    }
}

__inline__ __device__ Cell* ObjectFactory::createCellFromTO(CollectionTO const& collectionTO, int cellIndex, Cell* cellArray, bool createIds)
{
    auto cellTO = collectionTO.cells[cellIndex];
    Cell** cellPointer = _data->objects.cells.getNewElement();
    Cell* cell = cellArray + cellIndex;
    *cellPointer = cell;

    changeCellFromTO(collectionTO, cellTO, cell, createIds);
    cell->id = createIds ? _data->numberGen1.createNewId() : cellTO.id;
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
    if (cellTO.hasGenome) {
        auto const& genome = collectionTO.genomes[cellTO.genomeNodeIndex];
        cell->genome = &_data->objects.heap.atType<Genome>(genome.genomeIndexOnGpu);
    } else {
        cell->genome = nullptr;
    }
    return cell;
}

__inline__ __device__ void ObjectFactory::changeCellFromTO(CollectionTO const& collectionTO, CellTO const& cellTO, Cell* cell, bool createIds)
{
    cell->id = cellTO.id;
    cell->pos = cellTO.pos;
    _map.correctPosition(cell->pos);
    cell->vel = cellTO.vel;
    cell->livingState = cellTO.livingState;
    cell->creatureId = cellTO.creatureId;
    cell->mutationId = cellTO.mutationId;
    cell->ancestorMutationId = cellTO.ancestorMutationId;
    cell->energy = cellTO.energy;
    cell->stiffness = cellTO.stiffness;
    cell->cellType = cellTO.cellType;
    cell->barrier = cellTO.barrier;
    cell->sticky = cellTO.sticky;
    cell->age = cellTO.age;
    cell->color = cellTO.color;
    cell->angleToFront = cellTO.angleToFront;
    cell->activationTime = cellTO.activationTime;
    cell->genomeComplexity = cellTO.genomeComplexity;
    cell->detectedByCreatureId = cellTO.detectedByCreatureId;
    cell->cellTypeUsed = cellTO.cellTypeUsed;
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
        copyDataToHeap(
            cellTO.cellTypeData.constructor.genomeSize,
            cellTO.cellTypeData.constructor.genomeDataIndex,
            collectionTO.heap,
            cell->cellTypeData.constructor.genomeSize,
            cell->cellTypeData.constructor.genome);
        cell->cellTypeData.constructor.numInheritedGenomeNodes = cellTO.cellTypeData.constructor.numInheritedGenomeNodes;
        cell->cellTypeData.constructor.lastConstructedCellId = createIds ? 0 : cellTO.cellTypeData.constructor.lastConstructedCellId;
        cell->cellTypeData.constructor.genomeCurrentNodeIndex = cellTO.cellTypeData.constructor.genomeCurrentNodeIndex;
        cell->cellTypeData.constructor.genomeCurrentRepetition = cellTO.cellTypeData.constructor.genomeCurrentRepetition;
        cell->cellTypeData.constructor.genomeCurrentBranch = cellTO.cellTypeData.constructor.genomeCurrentBranch;
        cell->cellTypeData.constructor.offspringCreatureId = cellTO.cellTypeData.constructor.offspringCreatureId;
        cell->cellTypeData.constructor.offspringMutationId = cellTO.cellTypeData.constructor.offspringMutationId;
        cell->cellTypeData.constructor.genomeGeneration = cellTO.cellTypeData.constructor.genomeGeneration;
        cell->cellTypeData.constructor.constructionAngle1 = cellTO.cellTypeData.constructor.constructionAngle1;
        cell->cellTypeData.constructor.constructionAngle2 = cellTO.cellTypeData.constructor.constructionAngle2;
        cell->cellTypeData.constructor.isReady = true;
    } break;
    case CellType_Sensor: {
        cell->cellTypeData.sensor.autoTriggerInterval = cellTO.cellTypeData.sensor.autoTriggerInterval;
        cell->cellTypeData.sensor.minDensity = cellTO.cellTypeData.sensor.minDensity;
        cell->cellTypeData.sensor.minRange = cellTO.cellTypeData.sensor.minRange;
        cell->cellTypeData.sensor.maxRange = cellTO.cellTypeData.sensor.maxRange;
        cell->cellTypeData.sensor.restrictToColor = cellTO.cellTypeData.sensor.restrictToColor;
        cell->cellTypeData.sensor.restrictToMutants = cellTO.cellTypeData.sensor.restrictToMutants;
    } break;
    case CellType_Oscillator: {
        cell->cellTypeData.oscillator.autoTriggerInterval = cellTO.cellTypeData.oscillator.autoTriggerInterval;
        cell->cellTypeData.oscillator.alternationInterval = cellTO.cellTypeData.oscillator.alternationInterval;
        cell->cellTypeData.oscillator.numPulses = cellTO.cellTypeData.oscillator.numPulses;
    } break;
    case CellType_Attacker: {
    } break;
    case CellType_Injector: {
        cell->cellTypeData.injector.mode = cellTO.cellTypeData.injector.mode;
        cell->cellTypeData.injector.counter = cellTO.cellTypeData.injector.counter;
        copyDataToHeap(
            cellTO.cellTypeData.injector.genomeSize,
            cellTO.cellTypeData.injector.genomeDataIndex,
            collectionTO.heap,
            cell->cellTypeData.injector.genomeSize,
            cell->cellTypeData.injector.genome);
        cell->cellTypeData.injector.genomeGeneration = cellTO.cellTypeData.injector.genomeGeneration;
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
        cell->cellTypeData.reconnector.restrictToMutants = cellTO.cellTypeData.reconnector.restrictToMutants;
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
    particle->id = _data->numberGen1.createNewId();
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

    cell->id = _data->numberGen1.createNewId();
    cell->pos = pos;
    cell->vel = vel;
    cell->energy = energy;
    cell->stiffness = _data->numberGen1.random();
    cell->numConnections = 0;
    cell->livingState = LivingState_Ready;
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
    cell->genomeComplexity = 0;
    cell->signalRoutingRestriction.active = false;
    cell->signalRelaxationTime = 0;
    cell->signal.active = false;
    cell->density = 1.0f;
    cell->creatureId = 0;
    cell->mutationId = 0;
    cell->ancestorMutationId = 0;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->cellTypeUsed = CellTriggered_No;
    cell->genomeNodeIndex = 0;
    cell->cellType = CellType_Free;
    cell->neuralNetwork = nullptr;

    return cell;
}

__inline__ __device__ Cell* ObjectFactory::createCell(uint64_t& cellPointerIndex)
{
    auto cell = _data->objects.heap.getTypedSubArray<Cell>(1);
    auto cellPointer = _data->objects.cells.getNewElement(&cellPointerIndex);
    *cellPointer = cell;

    cell->id = _data->numberGen1.createNewId();
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
    cell->cellTypeUsed = CellTriggered_No;
    return cell;
}
