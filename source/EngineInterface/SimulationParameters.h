#pragma once

#include <cstdint>
#include <cstring>

#include "CellTypeConstants.h"
#include "Features.h"
#include "Motion.h"
#include "RadiationSource.h"
#include "SimulationParametersTypes.h"
#include "SimulationParametersZone.h"
#include "SimulationParametersZoneValues.h"

/**
 * NOTE: header is also included in kernel code
 */
struct SimulationParameters
{
    // Expert feature flags
    Features features;

    // Particle sources
    int numRadiationSources = 0;
    RadiationSource radiationSource[MAX_RADIATION_SOURCES];

    // Spots
    int numZones = 0;
    SimulationParametersZone zone[MAX_ZONES];

    // General
    Char64 projectName = "<unnamed>";

    // Visualization
    uint32_t backgroundColor = 0x1b0000;
    CellColoring primaryCellColoring = CellColoring_CellColor;
    CellType highlightedCellType = CellType_Constructor;
    float cellRadius = 0.25f;
    float zoomLevelForNeuronVisualization = 2.0f;
    bool attackVisualization = false;
    bool muscleMovementVisualization = false;
    bool borderlessRendering = false;
    bool gridLines = false;
    bool markReferenceDomain = true;
    bool showRadiationSources = true;

    // Numerics
    float timestepSize = 1.0f;

    // Physics: Motion
    Motion motionData;
    float innerFriction = 0.3f;

    // Physics: Thresholds
    float maxVelocity = 2.0f;
    float minCellDistance = 0.3f;
    static float constexpr maxForceDecayProbability = 0.2f;

    // Physics: Binding
    ColorVector<float> maxBindingDistance = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};

    // Physics: Radiation
    bool baseStrengthRatioPinned = false;
    ColorVector<int> radiationType1_minimumAge = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> radiationType2_strength = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> radiationType2_energyThreshold = {500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f};
    ColorVector<float> particleSplitEnergy = {
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value};
    bool particleTransformationAllowed = false;
    static float constexpr radiationProbability = 0.03f;
    static float constexpr radiationVelocityMultiplier = 1.0f;
    static float constexpr radiationVelocityPerturbation = 0.5f;

    // Cell life cycle
    ColorVector<int> maxCellAge = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    ColorVector<float> normalCellEnergy = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    CellDeathConsquences cellDeathConsequences = CellDeathConsquences_DetachedPartsDie;

    // Mutation
    ColorMatrix<bool> copyMutationColorTransitions = {
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true}};
    bool copyMutationPreventDepthIncrease = false;
    bool copyMutationSelfReplication = false;

    // Cell type: Attacker
    ColorVector<float> attackerStrength = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    ColorVector<float> attackerRadius = {1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f};
    bool attackerDestroyCells = true;

    // Cell type: Constructor
    ColorVector<float> constructorConnectingCellDistance = {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f};
    bool constructorCompletenessCheck = false;
    static float constexpr constructorAdditionalOffspringDistance = 0.8f;

    // Cell type: Defender
    ColorVector<float> defenderAntiAttackerStrength = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    ColorVector<float> defenderAntiInjectorStrength = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    // Cell type: Injector
    ColorVector<float> injectorInjectionRadius = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    ColorMatrix<int> injectorInjectionTime = {
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3}};

    // Cell type: Muscle
    ColorVector<float> muscleEnergyCost = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> muscleMovementAcceleration = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> muscleCrawlingAcceleration = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> muscleBendingAcceleration = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    static float constexpr cellTypeMuscleThreshold = 0.2f;
    static int constexpr cellTypeMuscleActivationCountdown = 10;

    // Cell type: Sensor
    ColorVector<float> sensorRadius = {255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f};

    // Cell type: Transmitter
    ColorVector<float> transmitterEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    ColorVector<float> transmitterEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    bool transmitterEnergyDistributionSameCreature = true;

    // Cell type: Reconnector
    ColorVector<float> reconnectorRadius = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    // Cell type: Detonator
    ColorVector<float> detonatorRadius = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    ColorVector<float> detonatorChainExplosionProbability = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Expert settings: Advanced absorption control
    ColorVector<float> radiationAbsorptionLowConnectionPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationAbsorptionHighVelocityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Expert settings: Advanced attacker control
    ColorMatrix<float> attackerSameMutantPenalty = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    ColorVector<float> attackerSensorDetectionFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    static float constexpr attackerColorInhomogeneityFactor = 1.0f;

    // Expert settings: Cell age limiter
    bool maxAgeForInactiveCellsActivated = false;  // Candidate for deletion
    bool freeCellMaxAgeActivated = false;
    ColorVector<int> freeCellMaxAge = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    bool resetCellAgeAfterActivation = false;   // Candidate for deletion
    bool maxCellAgeBalancerActivated = false;
    int maxCellAgeBalancerInterval = 10000;

    // Expert settings: Cell glow
    CellColoring cellGlowColoring = CellColoring_CellColor;
    float cellGlowRadius = 4.0f;
    float cellGlowStrength = 0.1f;

    // Expert settings: Customize deletion mutations setting
    int cellCopyMutationDeletionMinSize = 0;

    // Expert settings: Customize neuron mutations setting
    float cellCopyMutationNeuronDataWeight = 0.2f;
    float cellCopyMutationNeuronDataBias = 0.2f;
    float cellCopyMutationNeuronDataActivationFunction = 0.05f;
    float cellCopyMutationNeuronDataReinforcement = 1.05f;
    float cellCopyMutationNeuronDataDamping = 1.05f;
    float cellCopyMutationNeuronDataOffset = 0.05f;

    // Expert settings: External energy settings
    float externalEnergy = 0.0f;
    ColorVector<float> externalEnergyInflowFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> externalEnergyConditionalInflowFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool externalEnergyInflowOnlyForNonSelfReplicators = false;
    ColorVector<float> externalEnergyBackflowFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float externalEnergyBackflowLimit = Infinity<float>::value;

    // Expert settings: Genome complexity measurement
    ColorVector<float> genomeComplexitySizeFactor = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> genomeComplexityRamificationFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<int> genomeComplexityDepthLevel = {3, 3, 3, 3, 3, 3, 3};

    // All other parameters
    SimulationParametersZoneValues baseValues;

    bool operator==(SimulationParameters const&) const = default;
};
