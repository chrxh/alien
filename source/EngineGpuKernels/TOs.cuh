#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellFunctionConstants.h"

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
    int nameSize;
    uint64_t nameDataIndex;

    int descriptionSize;
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
    NeuronActivationFunction activationFunctions[MAX_CHANNELS];
};

struct TransmitterTO
{
    EnergyDistributionMode mode;
};

struct ConstructorTO
{
    int activationMode;  //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int constructionActivationTime;

    int genomeSize;
    uint64_t genomeDataIndex;
    int genomeGeneration;
    float constructionAngle1;
    float constructionAngle2;

    //process data
    uint64_t lastConstructedCellId;
    int genomeCurrentNodeIndex;
    int genomeCurrentRepetition;
    int offspringCreatureId;
    int offspringMutationId;
    uint32_t stateFlags;  //bit 0: isConstructionBuilt, bit 1: isInherited
};

struct SensorTO
{
    SensorMode mode;
    float angle;
    float minDensity;
    int color;
    int targetedCreatureId;

    //process data
    float memoryChannel1;
    float memoryChannel2;
    float memoryChannel3;
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
    int genomeSize;
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

struct ReconnectorTO
{
    int color;
};

struct DetonatorTO
{
    DetonatorState state;
    int countdown;
};

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
    ReconnectorTO reconnector;
    DetonatorTO detonator;
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
    int creatureId;
    int mutationId;

    //cell function
    int executionOrderNumber;
    int inputExecutionOrderNumber;
    bool outputBlocked;
    CellFunction cellFunction;
    CellFunctionTO cellFunctionData;
    ActivityTO activity;
    int activationTime;
    int genomeNumNodes;

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

