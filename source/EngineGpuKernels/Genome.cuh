#pragma once

#include <cstdint>

#include "EngineInterface/CellTypeConstants.h"

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
    EnergyDistributionMode mode;
};

struct SensorGenome
{
    uint8_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToMutants restrictToMutants;
};

struct ConstructorGenome
{
    int autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    int constructGeneIndex;
    int constructionActivationTime;
    float constructionAngle1;
    float constructionAngle2;
};

struct OscillatorGenome
{
    uint8_t autoTriggerInterval;
    uint8_t alternationInterval;  // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
};

struct AttackerGenome
{};

struct InjectorGenome
{
    InjectorMode mode;
};

struct AutoBendingGenome
{
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1
};

struct ManualBendingGenome
{
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1
};

struct AngleBendingGenome
{
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1
};

struct AutoCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;     // Between 0 and 1
};

struct ManualCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;     // Between 0 and 1
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

struct ReconnectorGenome
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToMutants restrictToMutants;
};

struct DetonatorGenome
{
    int32_t countdown;
};

union CellTypeDataGenome
{
    BaseGenome base;
    DepotGenome depot;
    ConstructorGenome constructor;
    SensorGenome sensor;
    OscillatorGenome oscillator;
    AttackerGenome attacker;
    InjectorGenome injector;
    MuscleGenome muscle;
    DefenderGenome defender;
    ReconnectorGenome reconnector;
    DetonatorGenome detonator;
};

struct SignalRoutingRestrictionGenome
{
    bool active;
    float baseAngle;
    float openingAngle;
};

struct Node
{
    float referenceAngle;
    int color;
    int numRequiredAdditionalConnections;

    NeuralNetworkGenome* neuralNetwork;
    CellTypeGenome cellType;
    CellTypeDataGenome cellTypeData;
    SignalRoutingRestrictionGenome signalRoutingRestriction;
};

struct Gene
{
    ConstructionShape shape;
    int numBranches;  // Between 1 and 6 in modulo
    bool separateConstruction;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numConcatenations;
    float concatenationAngle1;
    float concatenationAngle2;

    int numNodes;
    Node* nodes;
};

struct Genome
{
    float frontAngle;

    int numGenes;
    Gene* genes;

    // Temporary data
    uint64_t genomeIndex;
    static auto constexpr GenomeIndex_NotSet = 0xffffffffffffffff;
};
