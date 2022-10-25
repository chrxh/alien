#pragma once

#include "EngineInterface/Constants.h"
#include "EngineInterface/Enums.h"

#include "Base.cuh"
#include "TOs.cuh"
#include "Map.cuh"
#include "Math.cuh"
#include "Particle.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

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
    cell->underConstruction = cellTO.underConstruction;
    cell->inputBlocked = cellTO.inputBlocked;
    cell->outputBlocked = cellTO.outputBlocked;
    cell->maxConnections = cellTO.maxConnections;
    cell->energy = cellTO.energy;
    cell->cellFunction = cellTO.cellFunction;
    cell->barrier = cellTO.barrier;
    cell->age = cellTO.age;
    cell->color = cellTO.color;

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
            cellTO.cellFunctionData.neuron.weightsAndBiasSize,
            cellTO.cellFunctionData.neuron.weightsAndBiasDataIndex,
            dataTO.auxiliaryData,
            reinterpret_cast<uint8_t*&>(cell->cellFunctionData.neuron.neuronState));
    } break;
    case Enums::CellFunction_Transmitter: {
    } break;
    case Enums::CellFunction_Ribosome: {
        cell->cellFunctionData.ribosome.mode = cellTO.cellFunctionData.ribosome.mode;
        cell->cellFunctionData.ribosome.singleConstruction = cellTO.cellFunctionData.ribosome.singleConstruction;
        cell->cellFunctionData.ribosome.separateConstruction = cellTO.cellFunctionData.ribosome.separateConstruction;
        createAuxiliaryData(
            cellTO.cellFunctionData.ribosome.genomeSize,
            cellTO.cellFunctionData.ribosome.genomeDataIndex,
            dataTO.auxiliaryData,
            cell->cellFunctionData.ribosome.genomeSize,
            cell->cellFunctionData.ribosome.genome);
        cell->cellFunctionData.ribosome.currentGenomePos = cellTO.cellFunctionData.ribosome.currentGenomePos;
    } break;
    case Enums::CellFunction_Sensor: {
        cell->cellFunctionData.sensor.mode = cellTO.cellFunctionData.sensor.mode;
        cell->cellFunctionData.sensor.color = cellTO.cellFunctionData.sensor.color;
    } break;
    case Enums::CellFunction_Nerve: {
    } break;
    case Enums::CellFunction_Attacker: {
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
        printf("size: %llu\n", size);
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
    cell->maxConnections = _data->numberGen1.random(MAX_CELL_BONDS);
    cell->executionOrderNumber = _data->numberGen1.random(cudaSimulationParameters.cellMaxExecutionOrderNumber - 1);
    cell->numConnections = 0;
    cell->underConstruction = false;
    cell->locked = 0;
    cell->selected = 0;
    cell->color = 0;
    cell->metadata.nameSize = 0;
    cell->metadata.descriptionSize = 0;
    cell->barrier = false;
    cell->age = 0;

    cell->cellFunction = _data->numberGen1.random(Enums::CellFunction_Count - 1);
    return cell;
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
    return result;
}
