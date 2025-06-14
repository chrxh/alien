#pragma once

#include <cstdint>

#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/EngineConstants.h"

struct NeuralNetworkGenomeTO
{
    float weights[MAX_CHANNELS * MAX_CHANNELS];
    float biases[MAX_CHANNELS];
    ActivationFunction activationFunctions[MAX_CHANNELS];
};

struct BaseGenomeTO
{};

struct DepotGenomeTO
{
    EnergyDistributionMode mode;
};

struct SensorGenomeTO
{
    uint8_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToMutants restrictToMutants;
};

struct ConstructorGenomeTO
{
    int autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    int constructGeneIndex;
    int constructionActivationTime;
    float constructionAngle;
};

struct OscillatorGenomeTO
{
    uint8_t autoTriggerInterval;
    uint8_t alternationInterval;  // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
};

struct AttackerGenomeTO
{};

struct InjectorGenomeTO
{
    InjectorMode mode;
};

struct AutoBendingGenomeTO
{
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1
};

struct ManualBendingGenomeTO
{
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1
};

struct AngleBendingGenomeTO
{
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1
};

struct AutoCrawlingGenomeTO
{
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;     // Between 0 and 1
};

struct ManualCrawlingGenomeTO
{
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;     // Between 0 and 1
};

struct DirectMovementGenomeTO
{};

union MuscleModeGenomeTO
{
    AutoBendingGenomeTO autoBending;
    ManualBendingGenomeTO manualBending;
    AngleBendingGenomeTO angleBending;
    AutoCrawlingGenomeTO autoCrawling;
    ManualCrawlingGenomeTO manualCrawling;
    DirectMovementGenomeTO directMovement;
};

struct MuscleGenomeTO
{
    MuscleMode mode;
    MuscleModeGenomeTO modeData;
};

struct DefenderGenomeTO
{
    DefenderMode mode;
};

struct ReconnectorGenomeTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToMutants restrictToMutants;
};

struct DetonatorGenomeTO
{
    int32_t countdown;
};

union CellTypeDataGenomeTO
{
    BaseGenomeTO base;
    DepotGenomeTO depot;
    ConstructorGenomeTO constructor;
    SensorGenomeTO sensor;
    OscillatorGenomeTO oscillator;
    AttackerGenomeTO attacker;
    InjectorGenomeTO injector;
    MuscleGenomeTO muscle;
    DefenderGenomeTO defender;
    ReconnectorGenomeTO reconnector;
    DetonatorGenomeTO detonator;
};

struct SignalRoutingRestrictionGenomeTO
{
    bool active;
    float baseAngle;
    float openingAngle;
};


struct NodeTO
{
    float referenceAngle;
    int color;
    int numRequiredAdditionalConnections;

    uint64_t neuralNetworkDataIndex;
    CellTypeGenome cellType;
    CellTypeDataGenomeTO cellTypeData;
    SignalRoutingRestrictionGenomeTO signalRoutingRestriction;
};

struct GeneTO
{
    ConstructionShape shape;
    int numBranches;    // Between 1 and 6 in modulo
    bool separateConstruction;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numConcatenations;

    int numNodes;
    uint64_t nodeArrayIndex;
};

struct GenomeTO
{
    float frontAngle;

    int numGenes;
    uint64_t geneArrayIndex;

    // Temporary data
    uint64_t genomeIndexOnGpu;
};
