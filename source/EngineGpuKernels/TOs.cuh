#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "EngineInterface/Constants.h"
#include "EngineInterface/Enums.h"

struct ParticleTO
{
	uint64_t id;
	float energy;
	float2 pos;
	float2 vel;
    int color;

	int selected;
};

struct CellMetadataTO
{
    uint64_t nameSize;
    uint64_t nameDataIndex;

    uint64_t descriptionSize;
    uint64_t descriptionDataIndex;
};

struct ConnectionTO
{
    int cellIndex;
    float distance;
    float angleFromPrevious;
};

struct ActivityTO
{
    float channels[MAX_CHANNELS];
};

struct NeuronTO
{
    uint64_t weightsAndBiasDataIndex;   //size is always: sizeof(float) * MAX_CHANNELS * (MAX_CHANNELS + 1)
};

struct TransmitterTO
{};

struct ConstructorTO
{
    int mode;  //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    bool singleConstruction;
    bool separateConstruction;
    bool makeSticky;
    int angleAlignment;  //0: no alignment, 2: alignment to 180 deg, 3: alignment to 120 deg, ... up to 6
    uint64_t genomeSize;
    uint64_t genomeDataIndex;

    //process data
    uint64_t currentGenomePos;
};

struct SensorTO
{
    Enums::SensorMode mode;
    int color;
};

struct NerveTO
{};

struct AttackerTO
{};

struct InjectorTO
{
    uint64_t genomeSize;
    uint64_t genomeDataIndex;
};

struct MuscleTO
{};

struct PlaceHolderTO1
{};

struct PlaceHolderTO2
{};

union CellFunctionTO
{
    NeuronTO neuron;
    TransmitterTO transmitter;
    ConstructorTO constructor;
    SensorTO sensor;
    NerveTO nerve;
    AttackerTO attacker;
    InjectorTO injector;
    MuscleTO muscle;
    PlaceHolderTO1 placeHolder1;
    PlaceHolderTO2 placeHolder2;
};

struct CellTO
{
	uint64_t id;
    ConnectionTO connections[MAX_CELL_BONDS];

    float2 pos;
    float2 vel;
	float energy;
    int color;
    int maxConnections;
	int numConnections;
	int executionOrderNumber;
    bool barrier;
    int age;

    bool underConstruction;
    bool inputBlocked;
    bool outputBlocked;

    Enums::CellFunction cellFunction;
    CellFunctionTO cellFunctionData;
    ActivityTO activity;

    CellMetadataTO metadata;

    int selected;
};

struct DataTO
{
	uint64_t* numCells = nullptr;
	CellTO* cells = nullptr;
    uint64_t* numParticles = nullptr;
	ParticleTO* particles = nullptr;
    uint64_t* numAuxiliaryData = nullptr;
    uint8_t* auxiliaryData = nullptr;

	bool operator==(DataTO const& other) const
	{
		return numCells == other.numCells
			&& cells == other.cells
			&& numParticles == other.numParticles
			&& particles == other.particles
            && numAuxiliaryData == other.numAuxiliaryData
            && auxiliaryData == other.auxiliaryData;
	}
};

