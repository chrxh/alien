#pragma once

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellFunctionConstants.h"
#include "EngineInterface/GenomeConstants.h"

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "TOs.cuh"
#include "Map.cuh"
#include "Object.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"

class ObjectFactory
{
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Particle* createParticleFromTO(ParticleTO const& particleTO, bool createIds);
    __inline__ __device__ Cell* createCellFromTO(DataTO const& dataTO, int targetIndex, CellTO const& cellTO, Cell* cellArray, bool createIds);
    __inline__ __device__ void changeCellFromTO(DataTO const& dataTO, CellTO const& cellTO, Cell* cell, bool createIds);
    __inline__ __device__ void changeParticleFromTO(ParticleTO const& particleTO, Particle* particle);
    __inline__ __device__ Particle* createParticle(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Cell* createRandomCell(float energy, float2 const& pos, float2 const& vel);
    __inline__ __device__ Cell* createCell(uint64_t& cellPointerIndex);

private:
    template<typename T>
    __inline__ __device__ void createAuxiliaryData(T sourceSize, uint64_t sourceIndex, uint8_t* auxiliaryData, T& targetSize, uint8_t*& target);
    __inline__ __device__ void createAuxiliaryDataWithFixedSize(uint64_t size, uint64_t sourceIndex, uint8_t* auxiliaryData, uint8_t*& target);

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
    Particle** particlePointer = _data->objects.particlePointers.getNewElement();
    Particle* particle = _data->objects.particles.getNewElement();
    *particlePointer = particle;
    
    particle->id = createIds ? _data->numberGen1.createNewId() : particleTO.id;
    particle->absPos = particleTO.pos;
    _map.correctPosition(particle->absPos);
    particle->vel = particleTO.vel;
    particle->energy = particleTO.energy;
    particle->locked = 0;
    particle->selected = 0;
    particle->color = particleTO.color;
    particle->lastAbsorbedCell = nullptr;
    return particle;
}

__inline__ __device__ Cell* ObjectFactory::createCellFromTO(DataTO const& dataTO, int targetIndex, CellTO const& cellTO, Cell* cellTargetArray, bool createIds)
{
    Cell** cellPointer = _data->objects.cellPointers.getNewElement();
    Cell* cell = cellTargetArray + targetIndex;
    *cellPointer = cell;

    changeCellFromTO(dataTO, cellTO, cell, createIds);
    cell->id = createIds ? _data->numberGen1.createNewId() : cellTO.id;
    cell->locked = 0;
    cell->detached = 0;
    cell->selected = 0;
    cell->scheduledOperationIndex = -1;
    cell->numConnections = cellTO.numConnections;
    cell->event = CellEvent_No;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectingCell = cell->connections[i];
        connectingCell.cell = cellTargetArray + cellTO.connections[i].cellIndex;
        connectingCell.distance = cellTO.connections[i].distance;
        connectingCell.angleFromPrevious = cellTO.connections[i].angleFromPrevious;
    }
    cell->density = 1.0f;
    return cell;
}

__inline__ __device__ void ObjectFactory::changeCellFromTO(DataTO const& dataTO, CellTO const& cellTO, Cell* cell, bool createIds)
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
    cell->cellFunction = cellTO.cellFunction;
    cell->barrier = cellTO.barrier;
    cell->age = cellTO.age;
    cell->color = cellTO.color;
    cell->activationTime = cellTO.activationTime;
    cell->genomeComplexity = cellTO.genomeComplexity;
    cell->detectedByCreatureId = cellTO.detectedByCreatureId;
    cell->cellFunctionUsed = cellTO.cellFunctionUsed;

    createAuxiliaryData(cellTO.metadata.nameSize, cellTO.metadata.nameDataIndex, dataTO.auxiliaryData, cell->metadata.nameSize, cell->metadata.name);

    createAuxiliaryData(
        cellTO.metadata.descriptionSize,
        cellTO.metadata.descriptionDataIndex,
        dataTO.auxiliaryData,
        cell->metadata.descriptionSize,
        cell->metadata.description);

    cell->signalRoutingRestriction.active = cellTO.signalRoutingRestriction.active;
    cell->signalRoutingRestriction.baseAngle = cellTO.signalRoutingRestriction.baseAngle;
    cell->signalRoutingRestriction.openingAngle = cellTO.signalRoutingRestriction.openingAngle;

    cell->signal.active = cellTO.signal.active;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->signal.channels[i] = cellTO.signal.channels[i];
    }
    cell->signal.origin = cellTO.signal.origin;
    cell->signal.targetX = cellTO.signal.targetX;
    cell->signal.targetY = cellTO.signal.targetY;
    cell->signal.numPrevCells = cellTO.signal.numPrevCells;
    for (int i = 0; i < MAX_CELL_BONDS; ++i) {
        cell->signal.prevCellIds[i] = cellTO.signal.prevCellIds[i];
    }

    cell->cellFunction = cellTO.cellFunction;
    switch (cellTO.cellFunction) {
    case CellFunction_Neuron: {
        createAuxiliaryDataWithFixedSize(
            sizeof(NeuronFunction::NeuronState),
            cellTO.cellFunctionData.neuron.weightsAndBiasesDataIndex,
            dataTO.auxiliaryData,
            reinterpret_cast<uint8_t*&>(cell->cellFunctionData.neuron.neuronState));
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->cellFunctionData.neuron.activationFunctions[i] = cellTO.cellFunctionData.neuron.activationFunctions[i];
        }
    } break;
    case CellFunction_Transmitter: {
        cell->cellFunctionData.transmitter.mode = cellTO.cellFunctionData.transmitter.mode;
    } break;
    case CellFunction_Constructor: {
        cell->cellFunctionData.constructor.activationMode = cellTO.cellFunctionData.constructor.activationMode;
        cell->cellFunctionData.constructor.constructionActivationTime = cellTO.cellFunctionData.constructor.constructionActivationTime;
        createAuxiliaryData(
            cellTO.cellFunctionData.constructor.genomeSize,
            cellTO.cellFunctionData.constructor.genomeDataIndex,
            dataTO.auxiliaryData,
            cell->cellFunctionData.constructor.genomeSize,
            cell->cellFunctionData.constructor.genome);
        cell->cellFunctionData.constructor.numInheritedGenomeNodes = cellTO.cellFunctionData.constructor.numInheritedGenomeNodes;
        cell->cellFunctionData.constructor.lastConstructedCellId = createIds ? 0 : cellTO.cellFunctionData.constructor.lastConstructedCellId;
        cell->cellFunctionData.constructor.genomeCurrentNodeIndex = cellTO.cellFunctionData.constructor.genomeCurrentNodeIndex;
        cell->cellFunctionData.constructor.genomeCurrentRepetition = cellTO.cellFunctionData.constructor.genomeCurrentRepetition;
        cell->cellFunctionData.constructor.currentBranch = cellTO.cellFunctionData.constructor.currentBranch;
        cell->cellFunctionData.constructor.offspringCreatureId = cellTO.cellFunctionData.constructor.offspringCreatureId;
        cell->cellFunctionData.constructor.offspringMutationId = cellTO.cellFunctionData.constructor.offspringMutationId;
        cell->cellFunctionData.constructor.genomeGeneration = cellTO.cellFunctionData.constructor.genomeGeneration;
        cell->cellFunctionData.constructor.constructionAngle1 = cellTO.cellFunctionData.constructor.constructionAngle1;
        cell->cellFunctionData.constructor.constructionAngle2 = cellTO.cellFunctionData.constructor.constructionAngle2;
        cell->cellFunctionData.constructor.isReady = true;
    } break;
    case CellFunction_Sensor: {
        cell->cellFunctionData.sensor.minDensity = cellTO.cellFunctionData.sensor.minDensity;
        cell->cellFunctionData.sensor.minRange = cellTO.cellFunctionData.sensor.minRange;
        cell->cellFunctionData.sensor.maxRange = cellTO.cellFunctionData.sensor.maxRange;
        cell->cellFunctionData.sensor.restrictToColor = cellTO.cellFunctionData.sensor.restrictToColor;
        cell->cellFunctionData.sensor.restrictToMutants = cellTO.cellFunctionData.sensor.restrictToMutants;
        cell->cellFunctionData.sensor.memoryChannel1 = cellTO.cellFunctionData.sensor.memoryChannel1;
        cell->cellFunctionData.sensor.memoryChannel2 = cellTO.cellFunctionData.sensor.memoryChannel2;
        cell->cellFunctionData.sensor.memoryChannel3 = cellTO.cellFunctionData.sensor.memoryChannel3;
        cell->cellFunctionData.sensor.memoryTargetX = cellTO.cellFunctionData.sensor.memoryTargetX;
        cell->cellFunctionData.sensor.memoryTargetY = cellTO.cellFunctionData.sensor.memoryTargetY;
    } break;
    case CellFunction_Oscillator: {
        cell->cellFunctionData.oscillator.pulseMode = cellTO.cellFunctionData.oscillator.pulseMode;
        cell->cellFunctionData.oscillator.alternationMode = cellTO.cellFunctionData.oscillator.alternationMode;
    } break;
    case CellFunction_Attacker: {
        cell->cellFunctionData.attacker.mode = cellTO.cellFunctionData.attacker.mode;
    } break;
    case CellFunction_Injector: {
        cell->cellFunctionData.injector.mode = cellTO.cellFunctionData.injector.mode;
        cell->cellFunctionData.injector.counter = cellTO.cellFunctionData.injector.counter;
        createAuxiliaryData(
            cellTO.cellFunctionData.injector.genomeSize,
            cellTO.cellFunctionData.injector.genomeDataIndex,
            dataTO.auxiliaryData,
            cell->cellFunctionData.injector.genomeSize,
            cell->cellFunctionData.injector.genome);
        cell->cellFunctionData.injector.genomeGeneration = cellTO.cellFunctionData.injector.genomeGeneration;
    } break;
    case CellFunction_Muscle: {
        cell->cellFunctionData.muscle.mode = cellTO.cellFunctionData.muscle.mode;
        cell->cellFunctionData.muscle.lastBendingDirection = cellTO.cellFunctionData.muscle.lastBendingDirection;
        cell->cellFunctionData.muscle.lastBendingSourceIndex = cellTO.cellFunctionData.muscle.lastBendingSourceIndex;
        cell->cellFunctionData.muscle.consecutiveBendingAngle = cellTO.cellFunctionData.muscle.consecutiveBendingAngle;
        cell->cellFunctionData.muscle.lastMovementX = cellTO.cellFunctionData.muscle.lastMovementX;
        cell->cellFunctionData.muscle.lastMovementY = cellTO.cellFunctionData.muscle.lastMovementY;
    } break;
    case CellFunction_Defender: {
        cell->cellFunctionData.defender.mode = cellTO.cellFunctionData.defender.mode;
    } break;
    case CellFunction_Reconnector: {
        cell->cellFunctionData.reconnector.restrictToColor = cellTO.cellFunctionData.reconnector.restrictToColor;
        cell->cellFunctionData.reconnector.restrictToMutants = cellTO.cellFunctionData.reconnector.restrictToMutants;
    } break;
    case CellFunction_Detonator: {
        cell->cellFunctionData.detonator.state = cellTO.cellFunctionData.detonator.state;
        cell->cellFunctionData.detonator.countdown = cellTO.cellFunctionData.detonator.countdown;
    } break;
    }
}

__inline__ __device__ void ObjectFactory::changeParticleFromTO(ParticleTO const& particleTO, Particle* particle)
{
    particle->energy = particleTO.energy;
    particle->absPos = particleTO.pos;
    particle->color = particleTO.color;
}

template <typename T>
__inline__ __device__ void ObjectFactory::createAuxiliaryData(T sourceSize, uint64_t sourceIndex, uint8_t* auxiliaryData, T& targetSize, uint8_t*& target)
{
    targetSize = sourceSize;
    createAuxiliaryDataWithFixedSize(sourceSize, sourceIndex, auxiliaryData, target);
}

__inline__ __device__ void ObjectFactory::createAuxiliaryDataWithFixedSize(uint64_t size, uint64_t sourceIndex, uint8_t* auxiliaryData, uint8_t*& target)
{
    if (size > 0) {
        target = _data->objects.auxiliaryData.getAlignedSubArray(size);
        for (int i = 0; i < size; ++i) {
            target[i] = auxiliaryData[sourceIndex + i];
        }
    }
}

__inline__ __device__ Particle*
ObjectFactory::createParticle(float energy, float2 const& pos, float2 const& vel, int color)
{
    Particle** particlePointer = _data->objects.particlePointers.getNewElement();
    Particle* particle = _data->objects.particles.getNewElement();
    *particlePointer = particle;
    particle->id = _data->numberGen1.createNewId();
    particle->selected = 0;
    particle->locked = 0;
    particle->energy = energy;
    particle->absPos = pos;
    particle->vel = vel;
    particle->color = color;
    particle->lastAbsorbedCell = nullptr;
    return particle;
}

__inline__ __device__ Cell* ObjectFactory::createRandomCell(float energy, float2 const& pos, float2 const& vel)
{
    auto cell = _data->objects.cells.getNewElement();
    auto cellPointers = _data->objects.cellPointers.getNewElement();
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
    cell->metadata.nameSize = 0;
    cell->metadata.descriptionSize = 0;
    cell->barrier = false;
    cell->age = 0;
    cell->activationTime = 0;
    cell->genomeComplexity = 0;
    cell->signalRoutingRestriction.active = false;
    cell->signal.active = false;
    cell->density = 1.0f;
    cell->creatureId = 0;
    cell->mutationId = Const::MutationIdForFreeCell;
    cell->ancestorMutationId = 0;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->cellFunctionUsed = CellFunctionUsed_No;

    if (cudaSimulationParameters.particleTransformationRandomCellFunction) {
        cell->cellFunction = _data->numberGen1.random(CellFunction_Count - 1);
        switch (cell->cellFunction) {
        case CellFunction_Neuron: {
            cell->cellFunctionData.neuron.neuronState =
                reinterpret_cast<NeuronFunction::NeuronState*>(_data->objects.auxiliaryData.getAlignedSubArray(sizeof(NeuronFunction::NeuronState)));
            for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
                cell->cellFunctionData.neuron.neuronState->weights[i] = _data->numberGen1.random(2.0f) - 1.0f;
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->cellFunctionData.neuron.neuronState->biases[i] = _data->numberGen1.random(2.0f) - 1.0f;
                cell->cellFunctionData.neuron.activationFunctions[i] = NeuronActivationFunction_Sigmoid;
            }
        } break;
        case CellFunction_Transmitter: {
            cell->cellFunctionData.transmitter.mode = _data->numberGen1.random(EnergyDistributionMode_Count - 1);
        } break;
        case CellFunction_Constructor: {
            if (_data->numberGen1.randomBool()) {
                cell->cellFunctionData.constructor.activationMode = 0;
            } else {
                cell->cellFunctionData.constructor.activationMode = _data->numberGen1.random(50);
            }
            cell->cellFunctionData.constructor.constructionActivationTime = _data->numberGen1.random(10000);
            cell->cellFunctionData.constructor.genomeSize = Const::GenomeHeaderSize;
            cell->cellFunctionData.constructor.numInheritedGenomeNodes = 0;
            cell->cellFunctionData.constructor.genome = _data->objects.auxiliaryData.getAlignedSubArray(cell->cellFunctionData.constructor.genomeSize);
            auto& genome = cell->cellFunctionData.constructor.genome;
            for (int i = 0; i < cell->cellFunctionData.constructor.genomeSize; ++i) {
                genome[i] = _data->numberGen1.randomByte();
            }
            cell->cellFunctionData.constructor.lastConstructedCellId = 0;
            cell->cellFunctionData.constructor.genomeCurrentNodeIndex = 0;
            cell->cellFunctionData.constructor.genomeCurrentRepetition = 0;
            cell->cellFunctionData.constructor.currentBranch = 0;
            cell->cellFunctionData.constructor.genomeGeneration = 0;
            cell->cellFunctionData.constructor.constructionAngle1 = 0;
            cell->cellFunctionData.constructor.constructionAngle2 = 0;
            cell->cellFunctionData.constructor.isReady = true;
        } break;
        case CellFunction_Sensor: {
            cell->cellFunctionData.sensor.minDensity = _data->numberGen1.random(1.0f);
            cell->cellFunctionData.sensor.minRange = -1;
            cell->cellFunctionData.sensor.maxRange = -1;
            cell->cellFunctionData.sensor.restrictToColor = _data->numberGen1.randomBool() ? _data->numberGen1.random(MAX_COLORS - 1) : 255;
            cell->cellFunctionData.sensor.restrictToMutants = static_cast<uint8_t>(_data->numberGen1.random(SensorRestrictToMutants_Count - 1));
            cell->cellFunctionData.sensor.memoryChannel1 = 0;
            cell->cellFunctionData.sensor.memoryChannel2 = 0;
            cell->cellFunctionData.sensor.memoryChannel3 = 0;
            cell->cellFunctionData.sensor.memoryTargetX = 0;
            cell->cellFunctionData.sensor.memoryTargetY = 0;
        } break;
        case CellFunction_Oscillator: {
        } break;
        case CellFunction_Attacker: {
            cell->cellFunctionData.attacker.mode = _data->numberGen1.random(EnergyDistributionMode_Count - 1);
        } break;
        case CellFunction_Injector: {
            cell->cellFunctionData.injector.mode = _data->numberGen1.random(InjectorMode_Count - 1);
            cell->cellFunctionData.injector.counter = 0;
            cell->cellFunctionData.injector.genomeSize = _data->numberGen1.random(cudaSimulationParameters.particleTransformationMaxGenomeSize);
            cell->cellFunctionData.injector.genome = _data->objects.auxiliaryData.getAlignedSubArray(cell->cellFunctionData.injector.genomeSize);
            auto& genome = cell->cellFunctionData.injector.genome;
            for (int i = 0; i < cell->cellFunctionData.injector.genomeSize; ++i) {
                genome[i] = _data->numberGen1.randomByte();
            }
            cell->cellFunctionData.injector.genomeGeneration = 0;
        } break;
        case CellFunction_Muscle: {
            cell->cellFunctionData.muscle.mode = _data->numberGen1.random(MuscleMode_Count - 1);
            cell->cellFunctionData.muscle.lastBendingDirection = MuscleBendingDirection_None;
            cell->cellFunctionData.muscle.lastBendingSourceIndex = 0;
            cell->cellFunctionData.muscle.consecutiveBendingAngle = 0;
            cell->cellFunctionData.muscle.lastMovementX = 0;
            cell->cellFunctionData.muscle.lastMovementY = 0;
        } break;
        case CellFunction_Defender: {
            cell->cellFunctionData.defender.mode = _data->numberGen1.random(DefenderMode_Count - 1);
        } break;
        case CellFunction_Reconnector: {
            cell->cellFunctionData.reconnector.restrictToColor = _data->numberGen1.randomBool() ? _data->numberGen1.random(MAX_COLORS - 1) : 255;
            cell->cellFunctionData.reconnector.restrictToMutants = static_cast<uint8_t>(_data->numberGen1.random(ReconnectorRestrictToMutants_Count - 1));
        } break;
        case CellFunction_Detonator: {
            cell->cellFunctionData.detonator.state = DetonatorState_Ready;
            cell->cellFunctionData.detonator.countdown = 10;
        } break;
        }

    } else {
        cell->cellFunction = CellFunction_None;
    }
    return cell;
}

__inline__ __device__ Cell* ObjectFactory::createCell(uint64_t& cellPointerIndex)
{
    auto cell = _data->objects.cells.getNewElement();
    auto cellPointer = _data->objects.cellPointers.getNewElement(&cellPointerIndex);
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
    cell->barrier = 0;
    cell->age = 0;
    cell->vel = {0, 0};
    cell->activationTime = 0;
    cell->signalRoutingRestriction.active = false;
    cell->signal.active = false;
    cell->density = 1.0f;
    cell->detectedByCreatureId = 0;
    cell->event = CellEvent_No;
    cell->cellFunctionUsed = CellFunctionUsed_No;
    return cell;
}
