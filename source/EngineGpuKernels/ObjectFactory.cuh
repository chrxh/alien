#pragma once

#include "EngineInterface/Constants.h"
#include "EngineInterface/CellFunctionEnums.h"

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "TOs.cuh"
#include "Map.cuh"
#include "Particle.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"

class ObjectFactory
{
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Particle* createParticleFromTO(ParticleTO const& particleTO, bool createIds);
    __inline__ __device__ Cell* createCellFromTO(DataTO const& dataTO, int targetIndex, CellTO const& cellTO, Cell* cellArray, bool createIds);
    __inline__ __device__ void changeCellFromTO(DataTO const& dataTO, CellTO const& cellTO, Cell* cell);
    __inline__ __device__ void changeParticleFromTO(ParticleTO const& particleTO, Particle* particle);
    __inline__ __device__ Particle* createParticle(float energy, float2 const& pos, float2 const& vel, int color);
    __inline__ __device__ Cell* createRandomCell(float energy, float2 const& pos, float2 const& vel);
    __inline__ __device__ Cell* createCell();

private:
    __inline__ __device__ void createAuxiliaryData(uint64_t sourceSize, uint64_t sourceIndex, uint8_t* auxiliaryData, uint64_t& targetSize, uint8_t*& target);
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
    
    particle->id = createIds ? _data->numberGen1.createNewId_kernel() : particleTO.id;
    particle->absPos = particleTO.pos;
    _map.correctPosition(particle->absPos);
    particle->vel = particleTO.vel;
    particle->energy = particleTO.energy;
    particle->locked = 0;
    particle->selected = 0;
    particle->color = particleTO.color;
    return particle;
}

__inline__ __device__ Cell* ObjectFactory::createCellFromTO(DataTO const& dataTO, int targetIndex, CellTO const& cellTO, Cell* cellTargetArray, bool createIds)
{
    Cell** cellPointer = _data->objects.cellPointers.getNewElement();
    Cell* cell = cellTargetArray + targetIndex;
    *cellPointer = cell;

    changeCellFromTO(dataTO, cellTO, cell);
    cell->id = createIds ? _data->numberGen1.createNewId_kernel() : cellTO.id;
    cell->locked = 0;
    cell->detached = 0;
    cell->selected = 0;
    cell->scheduledOperationIndex = -1;
    cell->numConnections = cellTO.numConnections;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectingCell = cell->connections[i];
        connectingCell.cell = cellTargetArray + cellTO.connections[i].cellIndex;
        connectingCell.distance = cellTO.connections[i].distance;
        connectingCell.angleFromPrevious = cellTO.connections[i].angleFromPrevious;
    }
    cell->density = 1.0f;
    return cell;
}

__inline__ __device__ void ObjectFactory::changeCellFromTO(DataTO const& dataTO, CellTO const& cellTO, Cell* cell)
{
    cell->id = cellTO.id;
    cell->absPos = cellTO.pos;
    _map.correctPosition(cell->absPos);
    cell->vel = cellTO.vel;
    cell->executionOrderNumber = cellTO.executionOrderNumber;
    cell->livingState = cellTO.livingState;
    cell->constructionId = cellTO.constructionId;
    cell->inputExecutionOrderNumber = cellTO.inputExecutionOrderNumber;
    cell->outputBlocked = cellTO.outputBlocked;
    cell->maxConnections = cellTO.maxConnections;
    cell->energy = cellTO.energy;
    cell->stiffness = cellTO.stiffness;
    cell->cellFunction = cellTO.cellFunction;
    cell->barrier = cellTO.barrier;
    cell->age = cellTO.age;
    cell->color = cellTO.color;
    cell->activationTime = cellTO.activationTime;

    createAuxiliaryData(cellTO.metadata.nameSize, cellTO.metadata.nameDataIndex, dataTO.auxiliaryData, cell->metadata.nameSize, cell->metadata.name);

    createAuxiliaryData(
        cellTO.metadata.descriptionSize,
        cellTO.metadata.descriptionDataIndex,
        dataTO.auxiliaryData,
        cell->metadata.descriptionSize,
        cell->metadata.description);

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->activity.channels[i] = cellTO.activity.channels[i];
    }

    cell->cellFunction = cellTO.cellFunction;
    switch (cellTO.cellFunction) {
    case CellFunction_Neuron: {
        createAuxiliaryDataWithFixedSize(
            sizeof(NeuronFunction::NeuronState),
            cellTO.cellFunctionData.neuron.weightsAndBiasesDataIndex,
            dataTO.auxiliaryData,
            reinterpret_cast<uint8_t*&>(cell->cellFunctionData.neuron.neuronState));
    } break;
    case CellFunction_Transmitter: {
        cell->cellFunctionData.transmitter.mode = cellTO.cellFunctionData.transmitter.mode;
    } break;
    case CellFunction_Constructor: {
        cell->cellFunctionData.constructor.activationMode = cellTO.cellFunctionData.constructor.activationMode;
        cell->cellFunctionData.constructor.singleConstruction = cellTO.cellFunctionData.constructor.singleConstruction;
        cell->cellFunctionData.constructor.separateConstruction = cellTO.cellFunctionData.constructor.separateConstruction;
        cell->cellFunctionData.constructor.maxConnections = cellTO.cellFunctionData.constructor.maxConnections;
        cell->cellFunctionData.constructor.angleAlignment = cellTO.cellFunctionData.constructor.angleAlignment;
        cell->cellFunctionData.constructor.stiffness = cellTO.cellFunctionData.constructor.stiffness;
        cell->cellFunctionData.constructor.constructionActivationTime = cellTO.cellFunctionData.constructor.constructionActivationTime;
        createAuxiliaryData(
            cellTO.cellFunctionData.constructor.genomeSize % MAX_GENOME_BYTES,
            cellTO.cellFunctionData.constructor.genomeDataIndex,
            dataTO.auxiliaryData,
            cell->cellFunctionData.constructor.genomeSize,
            cell->cellFunctionData.constructor.genome);
        cell->cellFunctionData.constructor.currentGenomePos = cellTO.cellFunctionData.constructor.currentGenomePos;
    } break;
    case CellFunction_Sensor: {
        cell->cellFunctionData.sensor.mode = cellTO.cellFunctionData.sensor.mode;
        cell->cellFunctionData.sensor.angle = cellTO.cellFunctionData.sensor.angle;
        cell->cellFunctionData.sensor.minDensity = cellTO.cellFunctionData.sensor.minDensity;
        cell->cellFunctionData.sensor.color = cellTO.cellFunctionData.sensor.color;
    } break;
    case CellFunction_Nerve: {
        cell->cellFunctionData.nerve.pulseMode = cellTO.cellFunctionData.nerve.pulseMode;
        cell->cellFunctionData.nerve.alternationMode = cellTO.cellFunctionData.nerve.alternationMode;
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
    } break;
    case CellFunction_Muscle: {
        cell->cellFunctionData.muscle.mode = cellTO.cellFunctionData.muscle.mode;
    } break;
    case CellFunction_Defender: {
        cell->cellFunctionData.defender.mode = cellTO.cellFunctionData.defender.mode;
    } break;
    case CellFunction_Placeholder: {
    } break;
    }
}

__inline__ __device__ void ObjectFactory::changeParticleFromTO(ParticleTO const& particleTO, Particle* particle)
{
    particle->energy = particleTO.energy;
    particle->absPos = particleTO.pos;
    particle->color = particleTO.color;
}

__inline__ __device__ void
ObjectFactory::createAuxiliaryData(uint64_t sourceSize, uint64_t sourceIndex, uint8_t* auxiliaryData, uint64_t& targetSize, uint8_t*& target)
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
    particle->id = _data->numberGen1.createNewId_kernel();
    particle->selected = 0;
    particle->locked = 0;
    particle->energy = energy;
    particle->absPos = pos;
    particle->vel = vel;
    particle->color = color;
    return particle;
}

__inline__ __device__ Cell* ObjectFactory::createRandomCell(float energy, float2 const& pos, float2 const& vel)
{
    auto cell = _data->objects.cells.getNewElement();
    auto cellPointers = _data->objects.cellPointers.getNewElement();
    *cellPointers = cell;

    cell->id = _data->numberGen1.createNewId_kernel();
    cell->absPos = pos;
    cell->vel = vel;
    cell->energy = energy;
    cell->stiffness = _data->numberGen1.random();
    cell->maxConnections = _data->numberGen1.random(MAX_CELL_BONDS);
    cell->executionOrderNumber = _data->numberGen1.random(cudaSimulationParameters.cellNumExecutionOrderNumbers - 1);
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
    cell->inputExecutionOrderNumber = _data->numberGen1.random(cudaSimulationParameters.cellNumExecutionOrderNumbers - 1);
    cell->outputBlocked = _data->numberGen1.randomBool();
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->activity.channels[i] = 0;
    }
    cell->density = 1.0f;

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
            cell->cellFunctionData.constructor.singleConstruction = _data->numberGen1.randomBool();
            cell->cellFunctionData.constructor.separateConstruction = _data->numberGen1.randomBool();
            cell->cellFunctionData.constructor.maxConnections = _data->numberGen1.random(MAX_CELL_BONDS + 1) - 1;
            cell->cellFunctionData.constructor.angleAlignment = _data->numberGen1.random(ConstructorAngleAlignment_Count - 1);
            cell->cellFunctionData.constructor.stiffness = _data->numberGen1.random();
            cell->cellFunctionData.constructor.constructionActivationTime = _data->numberGen1.random(10000);
            cell->cellFunctionData.constructor.genomeSize = 0;
            //_data->numberGen1.random(cudaSimulationParameters.particleTransformationMaxGenomeSize);
            //cell->cellFunctionData.constructor.genome = _data->objects.auxiliaryData.getAlignedSubArray(cell->cellFunctionData.constructor.genomeSize);
            //auto& genome = cell->cellFunctionData.constructor.genome;
            //for (int i = 0; i < cell->cellFunctionData.constructor.genomeSize; ++i) {
            //    genome[i] = _data->numberGen1.randomByte();
            //}
            cell->cellFunctionData.constructor.currentGenomePos = 0;
        } break;
        case CellFunction_Sensor: {
            cell->cellFunctionData.sensor.mode = _data->numberGen1.random(SensorMode_Count - 1);
            cell->cellFunctionData.sensor.angle = _data->numberGen1.random(360.0f) - 180.0f;
            cell->cellFunctionData.sensor.minDensity = _data->numberGen1.random(1.0f);
            cell->cellFunctionData.sensor.color = _data->numberGen1.random(MAX_COLORS - 1);
        } break;
        case CellFunction_Nerve: {
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
        } break;
        case CellFunction_Muscle: {
            cell->cellFunctionData.muscle.mode = _data->numberGen1.random(MuscleMode_Count - 1);
        } break;
        case CellFunction_Defender: {
            cell->cellFunctionData.defender.mode = _data->numberGen1.random(DefenderMode_Count - 1);
        } break;
        case CellFunction_Placeholder: {
        } break;
        }

    } else {
        cell->cellFunction = CellFunction_None;
    }
    return cell;
}

__inline__ __device__ Cell* ObjectFactory::createCell()
{
    auto cell = _data->objects.cells.getNewElement();
    auto cellPointer = _data->objects.cellPointers.getNewElement();
    *cellPointer = cell;

    cell->id = _data->numberGen1.createNewId_kernel();
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
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->activity.channels[i] = 0;
    }
    cell->density = 1.0f;
    return cell;
}
