#pragma once

#include <cstdint>

#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/SimulationParametersTypes.h"

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
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToCreatures restrictToCreatures;
};

struct ConstructorGenomeTO
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t geneIndex;
    uint16_t constructionActivationTime;
    float constructionAngle;
};

struct GeneratorGenomeTO
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.
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
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualBendingGenomeTO
{
    float maxAngleDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct AngleBendingGenomeTO
{
    float maxAngleDeviation;  // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1
};

struct AutoCrawlingGenomeTO
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;     // Between 0 and 1
};

struct ManualCrawlingGenomeTO
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;     // Between 0 and 1
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
    ReconnectorRestrictToCreatures restrictToCreatures;
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
    GeneratorGenomeTO generator;
    AttackerGenomeTO attacker;
    InjectorGenomeTO injector;
    MuscleGenomeTO muscle;
    DefenderGenomeTO defender;
    ReconnectorGenomeTO reconnector;
    DetonatorGenomeTO detonator;
};

struct SignalRestrictionGenomeTO
{
    bool active;
    float baseAngle;
    float openingAngle;
};


struct NodeTO
{
    float referenceAngle;
    int color;
    int numAdditionalConnections;

    NeuralNetworkGenomeTO neuralNetwork;
    CellTypeGenome cellType;
    CellTypeDataGenomeTO cellTypeData;
    SignalRestrictionGenomeTO signalRestriction;
};

struct GeneTO
{
    Char64 name;
    ConstructorShape shape;
    bool separation;
    uint8_t numBranches;    // For separation = false
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numConcatenations;

    int numNodes;
    uint64_t nodeArrayIndex;
};

struct GenomeTO
{
    Char64 name;
    int numGenes;
    uint64_t geneArrayIndex;

    float frontAngle;
};
