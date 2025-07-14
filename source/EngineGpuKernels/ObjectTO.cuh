#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "GenomeTO.cuh"
#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/ArraySizesForTO.h"

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
    uint64_t cellIndex;
    static auto constexpr CellIndex_NotSet = 0x7fffffffffffffff;

    float distance;
    float angleFromPrevious;
};

struct NeuralNetworkTO
{
    float weights[MAX_CHANNELS * MAX_CHANNELS];
    float biases[MAX_CHANNELS];
    ActivationFunction activationFunctions[MAX_CHANNELS];
};

struct BaseTO
{
};

struct DepotTO
{
    EnergyDistributionMode mode;
};

struct ConstructorTO
{
    // Properties
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t constructionActivationTime;

    // Genome data
    uint16_t geneIndex;

    // Process data
    uint64_t lastConstructedCellId;
    uint16_t currentNodeIndex;
    uint16_t currentConcatenation;
    uint8_t currentBranch;
};

struct SensorTO
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToCreatures restrictToCreatures;
};

struct GeneratorTO
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    uint32_t numPulses;
};

struct AttackerTO
{
};

struct InjectorTO
{
    InjectorMode mode;
    uint32_t counter;
};

struct AutoBendingTO
{
    // Fixed data
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
    float lastActualAngle;
    bool forward;  // Current direction
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};

struct ManualBendingTO
{
    // Fixed data
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
    float lastActualAngle;
    float lastAngleDelta;
    bool impulseAlreadyApplied;
};

struct AngleBendingTO
{
    // Fixed data
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
};

struct AutoCrawlingTO
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialDistance;
    float lastActualDistance;
    bool forward;  // Current direction
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};


struct ManualCrawlingTO
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialDistance;
    float lastActualDistance;
    float lastDistanceDelta;
    bool impulseAlreadyApplied;
};

struct DirectMovementTO
{};

union MuscleModeTO
{
    AutoBendingTO autoBending;
    ManualBendingTO manualBending;
    AngleBendingTO angleBending;
    AutoCrawlingTO autoCrawling;
    ManualCrawlingTO manualCrawling;
    DirectMovementTO directMovement;
};

struct MuscleTO
{
    MuscleMode mode;
    MuscleModeTO modeData;

    // Additional rendering data
    float lastMovementX;
    float lastMovementY;
};

struct DefenderTO
{
    DefenderMode mode;
};

struct ReconnectorTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToCreatures restrictToCreatures;
};

struct DetonatorTO
{
    DetonatorState state;
    int32_t countdown;
};

union CellTypeDataTO
{
    BaseTO base;
    DepotTO depot;
    ConstructorTO constructor;
    SensorTO sensor;
    GeneratorTO generator;
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
};

struct CellTO
{
    // General
    uint64_t id;
    ConnectionTO connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
	float energy;
    float stiffness;
    uint8_t color;
    uint8_t numConnections;
    float angleToFront;
    bool barrier;
    bool sticky;
    uint32_t age;
    CellState cellState;

    // Creature data
    bool belongToCreature;
    uint64_t creatureIndex;
    uint16_t genomeNodeIndex;

    // Cell type data
    uint64_t neuralNetworkDataIndex;  // Not used for structure and base cells
    static auto constexpr NeuralNetworkDataIndex_NotSet = 0xffffffffffffffff;
    CellType cellType;
    CellTypeDataTO cellTypeData;
    SignalRoutingRestrictionTO signalRoutingRestriction;
    uint8_t signalRelaxationTime;
    SignalTO signal;
    uint32_t activationTime;
    uint16_t detectedByCreatureId;  // Only the first 16 bits from the creature id
    CellTriggered cellTriggered;

    // Misc
    CellMetadataTO metadata;

    // Editing data
    uint8_t selected;
};

struct CreatureTO
{
    uint64_t id;
    uint64_t ancestorId;

    uint32_t generation;
    uint32_t mutationId;
    float genomeComplexity;

    GenomeTO genome;

    // Temporary data
    uint64_t creatureIndexOnGpu;
};

struct CollectionTO
{
    ArraySizesForTO capacities;

	uint64_t* numCells = nullptr;
	CellTO* cells = nullptr;
    uint64_t* numParticles = nullptr;
	ParticleTO* particles = nullptr;
    uint64_t* numCreatures = nullptr;
    CreatureTO* creatures = nullptr;
    uint64_t* numGenes = nullptr;
    GeneTO* genes = nullptr;
    uint64_t* numNodes = nullptr;
    NodeTO* nodes = nullptr;
    uint64_t* heapSize = nullptr;
    uint8_t* heap = nullptr;

	bool operator==(CollectionTO const& other) const = default;
};

