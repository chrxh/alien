#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/ArraySizes.h"

struct ParticleTO
{
	uint64_t id;
	float energy;
	float2 pos;
	float2 vel;
    uint8_t color;

	uint8_t selected;
};

struct CellMetadataTO
{
    uint16_t nameSize;
    uint64_t nameDataIndex;

    uint16_t descriptionSize;
    uint64_t descriptionDataIndex;
};

struct ConnectionTO
{
    int cellIndex;
    float distance;
    float angleFromPrevious;
};

struct NeuronTO
{
    static auto constexpr NeuronDataSize = sizeof(float) * MAX_CHANNELS * (MAX_CHANNELS + 1) + MAX_CHANNELS;
    uint64_t neuronDataIndex;
};

struct TransmitterTO
{
    EnergyDistributionMode mode;
};

struct ConstructorTO
{
    uint32_t activationMode;  //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    uint32_t constructionActivationTime;

    uint16_t genomeSize;
    uint16_t numInheritedGenomeNodes;
    uint16_t origGenomeSize;
    uint64_t genomeDataIndex;
    uint32_t genomeGeneration;
    float constructionAngle1;
    float constructionAngle2;

    //process data
    uint64_t lastConstructedCellId;
    uint16_t genomeCurrentNodeIndex;
    uint16_t genomeCurrentRepetition;
    uint8_t currentBranch;
    uint32_t offspringCreatureId;
    uint32_t offspringMutationId;
};

struct SensorTO
{
    float minDensity;
    int8_t minRange;          //< 0 = no restriction
    int8_t maxRange;          //< 0 = no restriction
    uint8_t restrictToColor;  //0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToMutants restrictToMutants;

    //process data
    float memoryChannel1;
    float memoryChannel2;
    float memoryChannel3;
    float memoryTargetX;
    float memoryTargetY;
};

struct OscillatorTO
{
    uint8_t pulseMode;    //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    uint8_t alternationMode;  //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
};

struct AttackerTO
{
    EnergyDistributionMode mode;
};

struct InjectorTO
{
    InjectorMode mode;
    uint32_t counter;
    uint16_t genomeSize;
    uint64_t genomeDataIndex;
    uint32_t genomeGeneration;
};

struct MuscleTO
{
    MuscleMode mode;
    MuscleBendingDirection lastBendingDirection;
    uint8_t lastBendingSourceIndex;
    float consecutiveBendingAngle;

    //additional rendering data
    float lastMovementX;
    float lastMovementY;
};

struct DefenderTO
{
    DefenderMode mode;
};

struct ReconnectorTO
{
    uint8_t restrictToColor;  //0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToMutants restrictToMutants;
};

struct DetonatorTO
{
    DetonatorState state;
    int32_t countdown;
};

union CellTypeTO
{
    NeuronTO neuron;
    TransmitterTO transmitter;
    ConstructorTO constructor;
    SensorTO sensor;
    OscillatorTO oscillator;
    AttackerTO attacker;
    InjectorTO injector;
    MuscleTO muscle;
    DefenderTO defender;
    ReconnectorTO reconnector;
    DetonatorTO detonator;
};

struct SignalRoutingRestrictionTO
{
    bool active;
    float baseAngle;
    float openingAngle;
};

struct SignalTO
{
    bool active;
    float channels[MAX_CHANNELS];
    SignalOrigin origin;
    float targetX;
    float targetY;
    int numPrevCells;
    uint64_t prevCellIds[MAX_CELL_BONDS];
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
    uint8_t color;
    uint8_t numConnections;
    bool barrier;
    uint32_t age;
    LivingState livingState;
    uint32_t creatureId;
    uint32_t mutationId;
    uint8_t ancestorMutationId;  //only the first 8 bits from ancestor mutation id
    float genomeComplexity;
    uint16_t genomeNodeIndex;

    //cell type data
    CellType cellType;
    CellTypeTO cellTypeData;
    SignalRoutingRestrictionTO signalRoutingRestriction;
    SignalTO signal;
    uint32_t activationTime;
    uint16_t detectedByCreatureId;  //only the first 16 bits from the creature id
    CellTriggered cellTypeUsed;

    CellMetadataTO metadata;

    uint8_t selected;
};

struct DataTO
{
	uint64_t* numCells = nullptr;
	CellTO* cells = nullptr;
    uint64_t* numParticles = nullptr;
	ParticleTO* particles = nullptr;
    uint64_t* numAuxiliaryData = nullptr;
    uint8_t* auxiliaryData = nullptr;

    void init(ArraySizes arraySizes)
    {
        numCells = new uint64_t;
        numParticles = new uint64_t;
        numAuxiliaryData = new uint64_t;
        *numCells = 0;
        *numParticles = 0;
        *numAuxiliaryData = 0;
        cells = new CellTO[arraySizes.cellArraySize];
        particles = new ParticleTO[arraySizes.particleArraySize];
        auxiliaryData = new uint8_t[arraySizes.auxiliaryDataSize];
    }

    void destroy()
    {
        delete numCells;
        delete numParticles;
        delete numAuxiliaryData;
        delete[] cells;
        delete[] particles;
        delete[] auxiliaryData;
    }

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

