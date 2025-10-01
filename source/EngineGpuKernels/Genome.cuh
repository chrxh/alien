#pragma once

#include <cstdint>

#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/SimulationParametersTypes.h"

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
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToCreatures restrictToCreatures;
};

struct ConstructorGenome
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t geneIndex;
    uint16_t constructionActivationTime;
    float constructionAngle;
    bool provideEnergy;
};

struct GeneratorGenome
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.
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
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualBendingGenome
{
    float maxAngleDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct AngleBendingGenome
{
    float maxAngleDeviation;  // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1
};

struct AutoCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;     // Between 0 and 1
};

struct ManualCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;     // Between 0 and 1
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
    ReconnectorRestrictToCreatures restrictToCreatures;
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
    GeneratorGenome generator;
    AttackerGenome attacker;
    InjectorGenome injector;
    MuscleGenome muscle;
    DefenderGenome defender;
    ReconnectorGenome reconnector;
    DetonatorGenome detonator;
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
    uint8_t numBranches;    // For separation = false
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numConcatenations;

    int numNodes;
    Node* nodes;
};

struct Genome
{
    Char64 name;
    int numGenes;
    Gene* genes;

    float frontAngle;
};
