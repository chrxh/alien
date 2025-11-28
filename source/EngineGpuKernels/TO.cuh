#pragma once

#include <EngineInterface/ArraySizesForTO.h>
#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>

#include "GenomeTO.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

struct ParticleTO
{
    uint64_t id;
    float energy;
    float2 pos;
    float2 vel;
    uint8_t color;

    uint8_t selected;
};

struct ConnectionTO
{
    uint64_t cellIndex;  // May be invalid

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
{};

struct DepotTO
{
    EnergyDistributionMode mode;
};

struct ConstructorTO
{
    // Properties
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t constructionActivationTime;
    float constructionAngle;
    ProvideEnergy provideEnergy;

    // Genome data
    uint16_t geneIndex;

    // Process data
    uint64_t lastConstructedCellId;  // May be invalid
    uint16_t currentNodeIndex;
    uint16_t currentConcatenation;
    uint8_t currentBranch;
};

struct DetectEnergyTO
{
    float minDensity;
};

struct DetectStructureTO
{};

struct DetectFreeCellTO
{
    float minDensity;
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct DetectCreatureLastMatchTO
{
    uint64_t creatureId;
    float2 pos;
};

struct DetectCreatureTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    DetectCreatureLineageRestriction restrictToLineage;

    // Process data
    bool lastMatchAvailable;
    DetectCreatureLastMatchTO lastMatch;
};

union SensorModeTO
{
    DetectEnergyTO detectEnergy;
    DetectStructureTO detectStructure;
    DetectFreeCellTO detectFreeCell;
    DetectCreatureTO detectCreature;
};

struct SensorTO
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    SensorMode mode;
    SensorModeTO modeData;
    uint16_t minRange;
    uint16_t maxRange;
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
{};

struct InjectorTO
{
    InjectorMode mode;
    uint32_t counter;
};

struct AutoBendingTO
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    bool forward;        // Current direction
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};

struct ManualBendingTO
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    float lastAngleDelta;
    bool impulseAlreadyApplied;
};

struct AngleBendingTO
{
    // Fixed data
    float maxAngleDeviation;         // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
};

struct AutoCrawlingTO
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialDistance;  // May be invalid
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
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialDistance;  // May be invalid
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

struct SignalRestrictionTO
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
    float frontAngle;  // May be invalid
    bool fixed;
    bool sticky;
    uint32_t age;
    CellState cellState;

    // Creature/Genome data
    bool belongToCreature;
    uint64_t creatureIndex;
    uint16_t nodeIndex;
    uint16_t parentNodeIndex;
    uint16_t geneIndex;

    // Process data
    uint16_t detectedByCreatureId;  // Only the first 16 bits from the creature id
    uint32_t frontAngleId;
    bool headCell;

    // Cell type data
    uint64_t neuralNetworkDataIndex;  // May be invalid (not used for structure and base cells)
    CellType cellType;
    CellTypeDataTO cellTypeData;
    SignalTO signal;
    SignalRestrictionTO signalRestriction;
    SignalState signalState;
    uint32_t activationTime;
    CellTriggered cellTriggered;

    // Editing data
    uint8_t selected;
};

struct CreatureTO
{
    uint64_t id;
    uint64_t ancestorId;

    uint32_t generation;
    uint32_t lineageId;
    uint32_t numCells;

    uint64_t genomeArrayIndex;

    // Process data
    uint32_t frontAngleId;

    // Temporary data
    uint64_t creatureIndexOnGpu;
};

struct TO
{
    ArraySizesForTO capacities;

    uint64_t* numCells = nullptr;
    CellTO* cells = nullptr;
    uint64_t* numParticles = nullptr;
    ParticleTO* particles = nullptr;
    uint64_t* numCreatures = nullptr;
    CreatureTO* creatures = nullptr;
    uint64_t* numGenomes = nullptr;
    GenomeTO* genomes = nullptr;
    uint64_t* numGenes = nullptr;
    GeneTO* genes = nullptr;
    uint64_t* numNodes = nullptr;
    NodeTO* nodes = nullptr;
    uint64_t* heapSize = nullptr;
    uint8_t* heap = nullptr;

    bool operator==(TO const& other) const = default;
};
