#pragma once

#include <cstdint>
#include <cstring>

#include "CellTypeConstants.h"
#include "ExpertToggles.h"
#include "RadiationSource.h"
#include "SimulationParametersTypes.h"
#include "SimulationParametersZone.h"
#include "SimulationParametersZoneValues.h"

/**
 * NOTE: header is also included in kernel code
 */
struct SimulationParameters
{
    // Sizes
    BaseParameter<int> numZones = {0};
    BaseParameter<int> numRadiationSources = {0};

    // General
    BaseParameter<Char64> projectName = {"<unnamed>"};

    // Visualization
    BaseZoneParameter<FloatColorRGB> backgroundColor = {.baseValue = {0.0f, 0.0f, 0.106f}};
    BaseParameter<CellColoring> primaryCellColoring = {CellColoring_CellColor};
    BaseParameter<CellType> highlightedCellType = {CellType_Constructor};
    BaseParameter<float> cellRadius = {0.25f};
    BaseParameter<float> zoomLevelForNeuronVisualization = {2.0f};
    BaseParameter<bool> attackVisualization = {false};
    BaseParameter<bool> muscleMovementVisualization = {false};
    BaseParameter<bool> borderlessRendering = {false};
    BaseParameter<bool> gridLines = {false};
    BaseParameter<bool> markReferenceDomain = {true};
    BaseParameter<bool> showRadiationSources = {true};

    // Numerics
    BaseParameter<float> timestepSize = {1.0f};

    // Physics: Motion
    BaseParameter<MotionType> motionType = {MotionType_Fluid};
    BaseParameter<float> smoothingLength = {0.8f};       // for MotionType_Fluid
    BaseParameter<float> viscosityStrength = {0.1f};     // for MotionType_Fluid
    BaseParameter<float> pressureStrength = {0.1f};      // for MotionType_Fluid
    BaseParameter<float> maxCollisionDistance = {1.3f};  // for MotionType_Collision
    BaseParameter<float> repulsionStrength = {0.08f};    // for MotionType_Collision
    BaseZoneParameter<float> friction = {.baseValue = 0.01f};
    BaseParameter<float> innerFriction = {0.3f};
    BaseZoneParameter<float> rigidity = {.baseValue = 0.0f};

    // Physics: Thresholds
    BaseParameter<float> maxVelocity = {2.0f};
    BaseZoneParameter<ColorVector<float>> maxForce = {.baseValue = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f}};
    BaseParameter<float> minCellDistance = {0.3f};
    static float constexpr maxForceDecayProbability = 0.2f;

    // Physics: Binding
    BaseParameter<ColorVector<float>> maxBindingDistance = {{3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f}};
    BaseZoneParameter<float> cellFusionVelocity = {.baseValue = 2.0f};
    BaseZoneParameter<float> cellMaxBindingEnergy = {.baseValue = Infinity<float>::value};

    // Physics: Radiation
    BaseParameter<bool> relativeStrengthPinned = {false};
    BaseZoneParameter<ColorVector<float>> radiationAbsorption = {.baseValue = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseZoneParameter<ColorVector<float>> radiationType1_strength = {.baseValue = {0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f}};
    BaseParameter<ColorVector<int>> radiationType1_minimumAge = {{0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorVector<float>> radiationType2_strength = {{0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorVector<float>> radiationType2_energyThreshold = {500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f};
    BaseParameter<ColorVector<float>> particleSplitEnergy = {
        {Infinity<float>::value,
         Infinity<float>::value,
         Infinity<float>::value,
         Infinity<float>::value,
         Infinity<float>::value,
         Infinity<float>::value,
         Infinity<float>::value}};
    BaseParameter<bool> particleTransformationAllowed = {false};
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
    ColorMatrix<float> attackerSameMutantProtection = {
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
    bool maxAgeForInactiveCellsEnabled = false;  // Candidate for deletion
    bool freeCellMaxAgeEnabled = false;
    ColorVector<int> freeCellMaxAge = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    bool resetCellAgeAfterActivation = false;   // Candidate for deletion
    bool maxCellAgeBalancerEnabled = false;
    int maxCellAgeBalancerInterval = 10000;

    // Expert settings: Cell glow
    CellColoring cellGlowColoring = CellColoring_CellColor;
    float cellGlowRadius = 4.0f;
    ColorVector<float> cellGlowStrength = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

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


    // OLD
    // Expert settings toggles
    ExpertToggles expertToggles;

    // Particle sources
    RadiationSource radiationSource[MAX_RADIATION_SOURCES];

    // Spots
    SimulationParametersZone zone[MAX_ZONES];


    bool operator==(SimulationParameters const&) const = default;
};
