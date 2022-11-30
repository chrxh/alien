#pragma once

#include "EngineInterface/Constants.h"
#include "EngineInterface/Enums.h"

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
    cell->numConnections = cellTO.numConnections;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectingCell = cell->connections[i];
        connectingCell.cell = cellTargetArray + cellTO.connections[i].cellIndex;
        connectingCell.distance = cellTO.connections[i].distance;
        connectingCell.angleFromPrevious = cellTO.connections[i].angleFromPrevious;
    }
    return cell;
}

__inline__ __device__ void ObjectFactory::changeCellFromTO(DataTO const& dataTO, CellTO const& cellTO, Cell* cell)
{
    cell->id = cellTO.id;
    cell->absPos = cellTO.pos;
    _map.correctPosition(cell->absPos);
    cell->vel = cellTO.vel;
    cell->executionOrderNumber = cellTO.executionOrderNumber;
    cell->constructionState = cellTO.constructionState;
    cell->inputBlocked = cellTO.inputBlocked;
    cell->outputBlocked = cellTO.outputBlocked;
    cell->maxConnections = cellTO.maxConnections;
    cell->energy = cellTO.energy;
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
    case Enums::CellFunction_Neuron: {
        createAuxiliaryDataWithFixedSize(
            sizeof(NeuronFunction::NeuronState),
            cellTO.cellFunctionData.neuron.weightsAndBiasDataIndex,
            dataTO.auxiliaryData,
            reinterpret_cast<uint8_t*&>(cell->cellFunctionData.neuron.neuronState));
    } break;
    case Enums::CellFunction_Transmitter: {
        cell->cellFunctionData.transmitter.mode = cellTO.cellFunctionData.transmitter.mode;
    } break;
    case Enums::CellFunction_Constructor: {
        cell->cellFunctionData.constructor.mode = cellTO.cellFunctionData.constructor.mode;
        cell->cellFunctionData.constructor.singleConstruction = cellTO.cellFunctionData.constructor.singleConstruction;
        cell->cellFunctionData.constructor.separateConstruction = cellTO.cellFunctionData.constructor.separateConstruction;
        cell->cellFunctionData.constructor.adaptMaxConnections = cellTO.cellFunctionData.constructor.adaptMaxConnections;
        cell->cellFunctionData.constructor.angleAlignment = cellTO.cellFunctionData.constructor.angleAlignment;
        cell->cellFunctionData.constructor.constructionActivationTime = cellTO.cellFunctionData.constructor.constructionActivationTime;
        createAuxiliaryData(
            cellTO.cellFunctionData.constructor.genomeSize % MAX_GENOME_BYTES,
            cellTO.cellFunctionData.constructor.genomeDataIndex,
            dataTO.auxiliaryData,
            cell->cellFunctionData.constructor.genomeSize,
            cell->cellFunctionData.constructor.genome);
        cell->cellFunctionData.constructor.currentGenomePos = cellTO.cellFunctionData.constructor.currentGenomePos;
    } break;
    case Enums::CellFunction_Sensor: {
        cell->cellFunctionData.sensor.mode = cellTO.cellFunctionData.sensor.mode;
        cell->cellFunctionData.sensor.angle = cellTO.cellFunctionData.sensor.angle;
        cell->cellFunctionData.sensor.minDensity = cellTO.cellFunctionData.sensor.minDensity;
        cell->cellFunctionData.sensor.color = cellTO.cellFunctionData.sensor.color;
    } break;
    case Enums::CellFunction_Nerve: {
    } break;
    case Enums::CellFunction_Attacker: {
        cell->cellFunctionData.attacker.mode = cellTO.cellFunctionData.attacker.mode;
    } break;
    case Enums::CellFunction_Injector: {
        createAuxiliaryData(
            cellTO.cellFunctionData.injector.genomeSize,
            cellTO.cellFunctionData.injector.genomeDataIndex,
            dataTO.auxiliaryData,
            cell->cellFunctionData.injector.genomeSize,
            cell->cellFunctionData.injector.genome);
    } break;
    case Enums::CellFunction_Muscle: {
        cell->cellFunctionData.muscle.mode = cellTO.cellFunctionData.muscle.mode;
    } break;
    case Enums::CellFunction_Placeholder1: {
    } break;
    case Enums::CellFunction_Placeholder2: {
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
    auto result = _data->objects.cells.getNewElement();
    auto cellPointers = _data->objects.cellPointers.getNewElement();
    *cellPointers = result;

    result->id = _data->numberGen1.createNewId_kernel();
    result->absPos = pos;
    result->vel = vel;
    result->energy = energy;
    result->maxConnections = _data->numberGen1.random(MAX_CELL_BONDS);
    result->executionOrderNumber = _data->numberGen1.random(cudaSimulationParameters.cellMaxExecutionOrderNumbers - 1);
    result->numConnections = 0;
    result->constructionState = false;
    result->locked = 0;
    result->selected = 0;
    result->color = 0;
    result->metadata.nameSize = 0;
    result->metadata.descriptionSize = 0;
    result->barrier = false;
    result->age = 0;
    result->activationTime = 0;
    result->inputBlocked = _data->numberGen1.randomBool();
    result->outputBlocked = _data->numberGen1.randomBool();
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result->activity.channels[i] = 0;
    }

    if (cudaSimulationParameters.createRandomCellFunction) {
        result->cellFunction = _data->numberGen1.random(Enums::CellFunction_Count - 1);
        switch (result->cellFunction) {
        case Enums::CellFunction_Neuron: {
            result->cellFunctionData.neuron.neuronState =
                reinterpret_cast<NeuronFunction::NeuronState*>(_data->objects.auxiliaryData.getAlignedSubArray(sizeof(NeuronFunction::NeuronState)));
            for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
                result->cellFunctionData.neuron.neuronState->weights[i] = _data->numberGen1.random(2.0f) - 1.0f;
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                result->cellFunctionData.neuron.neuronState->bias[i] = _data->numberGen1.random(2.0f) - 1.0f;
            }
        } break;
        case Enums::CellFunction_Transmitter: {
            result->cellFunctionData.transmitter.mode = _data->numberGen1.random(Enums::EnergyDistributionMode_Count - 1);
        } break;
        case Enums::CellFunction_Constructor: {
            if (_data->numberGen1.randomBool()) {
                result->cellFunctionData.constructor.mode = 0;
            } else {
                result->cellFunctionData.constructor.mode = _data->numberGen1.random(50);
            }
            result->cellFunctionData.constructor.singleConstruction = _data->numberGen1.randomBool();
            result->cellFunctionData.constructor.separateConstruction = _data->numberGen1.randomBool();
            result->cellFunctionData.constructor.adaptMaxConnections = _data->numberGen1.randomBool();
            result->cellFunctionData.constructor.angleAlignment = _data->numberGen1.random(Enums::ConstructorAngleAlignment_Count - 1);
            result->cellFunctionData.constructor.constructionActivationTime = _data->numberGen1.random(10000);
            result->cellFunctionData.constructor.genomeSize = _data->numberGen1.random(cudaSimulationParameters.randomMaxGenomeSize);
            result->cellFunctionData.constructor.genome = _data->objects.auxiliaryData.getAlignedSubArray(result->cellFunctionData.constructor.genomeSize);
            auto& genome = result->cellFunctionData.constructor.genome;
            for (int i = 0; i < result->cellFunctionData.constructor.genomeSize; ++i) {
                genome[i] = _data->numberGen1.randomByte();
            }
            result->cellFunctionData.constructor.currentGenomePos = 0;
        } break;
        case Enums::CellFunction_Sensor: {
            result->cellFunctionData.sensor.mode = _data->numberGen1.random(Enums::SensorMode_Count - 1);
            result->cellFunctionData.sensor.angle = _data->numberGen1.random(360.0f) - 180.0f;
            result->cellFunctionData.sensor.minDensity = _data->numberGen1.random(1.0f);
            result->cellFunctionData.sensor.color = _data->numberGen1.random(MAX_COLORS - 1);
        } break;
        case Enums::CellFunction_Nerve: {
        } break;
        case Enums::CellFunction_Attacker: {
            result->cellFunctionData.attacker.mode = _data->numberGen1.random(Enums::EnergyDistributionMode_Count - 1);
        } break;
        case Enums::CellFunction_Injector: {
            result->cellFunctionData.injector.genomeSize = _data->numberGen1.random(cudaSimulationParameters.randomMaxGenomeSize);
            result->cellFunctionData.injector.genome = _data->objects.auxiliaryData.getAlignedSubArray(result->cellFunctionData.injector.genomeSize);
            auto& genome = result->cellFunctionData.injector.genome;
            for (int i = 0; i < result->cellFunctionData.injector.genomeSize; ++i) {
                genome[i] = _data->numberGen1.randomByte();
            }
        } break;
        case Enums::CellFunction_Muscle: {
            result->cellFunctionData.muscle.mode = _data->numberGen1.random(Enums::MuscleMode_Count - 1);
        } break;
        case Enums::CellFunction_Placeholder1: {
        } break;
        case Enums::CellFunction_Placeholder2: {
        } break;
        }

    } else {
        result->cellFunction = Enums::CellFunction_None;
    }
    return result;
}

__inline__ __device__ Cell* ObjectFactory::createCell()
{
    auto result = _data->objects.cells.getNewElement();
    auto cellPointer = _data->objects.cellPointers.getNewElement();
    *cellPointer = result;

    result->id = _data->numberGen1.createNewId_kernel();
    result->selected = 0;
    result->locked = 0;
    result->color = 0;
    result->metadata.nameSize = 0;
    result->metadata.descriptionSize = 0;
    result->barrier = 0;
    result->age = 0;
    result->vel = {0, 0};
    result->activationTime = 0;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result->activity.channels[i] = 0;
    }
    return result;
}
