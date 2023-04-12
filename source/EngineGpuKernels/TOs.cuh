#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "EngineInterface/Constants.h"
#include "EngineInterface/CellFunctionEnums.h"

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
    uint64_t weightsAndBiasesDataIndex;   //size is always: sizeof(float) * MAX_CHANNELS * (MAX_CHANNELS + 1)
};

struct TransmitterTO
{
    EnergyDistributionMode mode;
};

struct ConstructorTO
{
    int activationMode;  //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    bool singleConstruction;
    bool separateConstruction;
    int maxConnections;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    int constructionActivationTime;

    uint64_t genomeSize;
    uint64_t genomeDataIndex;
    int genomeGeneration;

    //process data
    uint64_t currentGenomePos;
};

struct SensorTO
{
    SensorMode mode;
    float angle;
    float minDensity;
    int color;
};

struct NerveTO
{
    int pulseMode;        //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int alternationMode;  //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
};

struct AttackerTO
{
    EnergyDistributionMode mode;
};

struct InjectorTO
{
    InjectorMode mode;
    int counter;
    uint64_t genomeSize;
    uint64_t genomeDataIndex;
    int genomeGeneration;
};

struct MuscleTO
{
    MuscleMode mode;
    MuscleBendingDirection lastBendingDirection;
    int lastBendingSourceIndex;
    float consecutiveBendingAngle;
};

struct DefenderTO
{
    DefenderMode mode;
};

struct PlaceHolderTO
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
    DefenderTO defender;
    PlaceHolderTO placeHolder;
};

struct CellTO
{
	uint64_t id;

    //general
    ConnectionTO connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
	float energy;
    float stiffness;
    int color;
    int maxConnections;
	int numConnections;
    bool barrier;
    int age;
    LivingState livingState;
    int constructionId;

    //cell function
    int executionOrderNumber;
    int inputExecutionOrderNumber;
    bool outputBlocked;
    CellFunction cellFunction;
    CellFunctionTO cellFunctionData;
    ActivityTO activity;
    int activationTime;

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

