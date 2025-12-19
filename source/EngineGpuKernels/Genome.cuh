#pragma once

#include <cstdint>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/SimulationParametersTypes.h>

struct NeuralNetworkGenome
{
    float weights[MAX_CHANNELS * MAX_CHANNELS];
    float biases[MAX_CHANNELS];
    ActivationFunction activationFunctions[MAX_CHANNELS];
};

struct BaseGenome
{};

struct DepotGenome
{
    float maxUsableEnergyForStorage;
    float initialStoredUsableEnergy;
};

struct TelemetryGenome
{};

struct DetectEnergyGenome
{
    float minDensity;
};

struct DetectStructureGenome
{};

struct DetectFreeCellGenome
{
    float minDensity;
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct DetectCreatureGenome
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    DetectCreatureLineageRestriction restrictToLineage;
};

union SensorModeGenome
{
    TelemetryGenome telemetry;
    DetectEnergyGenome detectEnergy;
    DetectStructureGenome detectStructure;
    DetectFreeCellGenome detectFreeCell;
    DetectCreatureGenome detectCreature;
};

struct SensorGenome
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    SensorMode mode;
    SensorModeGenome modeData;
    uint16_t minRange;
    uint16_t maxRange;
};

struct ConstructorGenome
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t geneIndex;
    uint16_t constructionActivationTime;
    float constructionAngle;
    ProvideEnergy provideEnergy;
};

struct GeneratorGenome
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.
};

struct AttackFreeCellGenome
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct AttackCreatureGenome
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union AttackerModeGenome
{
    AttackFreeCellGenome attackFreeCell;
    AttackCreatureGenome attackCreature;
};

struct AttackerGenome
{
    AttackerMode mode;
    AttackerModeGenome modeData;
};

struct InjectorGenome
{
    uint16_t geneIndex;
};

struct AutoBendingGenome
{
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualBendingGenome
{
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct AngleBendingGenome
{
    float maxAngleDeviation;         // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1
};

struct AutoCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct DirectMovementGenome
{};

union MuscleModeGenome
{
    AutoBendingGenome autoBending;
    ManualBendingGenome manualBending;
    AngleBendingGenome angleBending;
    AutoCrawlingGenome autoCrawling;
    ManualCrawlingGenome manualCrawling;
    DirectMovementGenome directMovement;
};

struct MuscleGenome
{
    MuscleMode mode;
    MuscleModeGenome modeData;
};

struct DefenderGenome
{
    DefenderMode mode;
};

struct ReconnectStructureGenome
{};

struct ReconnectFreeCellGenome
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct ReconnectCreatureGenome
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    ReconnectCreatureLineageRestriction restrictToLineage;
};

union ReconnectorModeGenome
{
    ReconnectStructureGenome reconnectStructure;
    ReconnectFreeCellGenome reconnectFreeCell;
    ReconnectCreatureGenome reconnectCreature;
};

struct ReconnectorGenome
{
    ReconnectorMode mode;
    ReconnectorModeGenome modeData;
};

struct DetonatorGenome
{
    int32_t countdown;
};

struct DigestorGenome
{
    float rawEnergyConductivity;  // Between 0 and 1
};

union CellTypeDataGenome
{
    BaseGenome base;
    DepotGenome depot;
    ConstructorGenome constructor;
    SensorGenome sensor;
    GeneratorGenome generator;
    AttackerGenome attacker;
    InjectorGenome injector;
    MuscleGenome muscle;
    DefenderGenome defender;
    ReconnectorGenome reconnector;
    DetonatorGenome detonator;
    DigestorGenome digestor;
};

struct SignalRestrictionGenome
{
    bool active;
    float baseAngle;
    float openingAngle;
};

struct Node
{
    float referenceAngle;
    int color;
    int numAdditionalConnections;

    NeuralNetworkGenome neuralNetwork;
    CellTypeGenome cellType;
    CellTypeDataGenome cellTypeData;
    SignalRestrictionGenome signalRestriction;
};

struct Gene
{
    Char64 name;
    ConstructorShape shape;
    bool separation;
    uint8_t numBranches;  // For separation = false
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numConcatenations;

    int numNodes;
    Node* nodes;
};

struct Genome
{
    uint64_t id;
    Char64 name;
    int numGenes;
    Gene* genes;

    float frontAngle;

    // Temporary data
    uint64_t genomeIndex;  // May be invalid
};
