#pragma once

#include <cstdint>
#include <cstring>

#include "CellTypeConstants.h"
#include "RadiationSource.h"
#include "SimulationParametersTypes.h"
#include "SimulationParametersZone.h"
#include "Base/Vector2D.h"

/**
 * NOTE: header is also included in kernel code
 */
struct SimulationParameters
{
    /**
     * Sizes
     */
    BaseParameter<int> numZones = {0};
    BaseParameter<int> numRadiationSources = {0};

    /**
     * Source parameters
     */


    /**
     * Base and Zone parameters
     */

    // General
    BaseParameter<Char64> projectName = {"<unnamed>"};
    ZoneParameter<Char64> zoneNames = {"<unnamed>"};

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

    // Location
    ZoneParameter<RealVector2D> zonePosition;

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
    PinBaseParameter relativeStrengthPinned = {false};
    ZoneParameter<bool> radiationDisableSources = {{false}};
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
    BaseParameter<ColorVector<int>> maxCellAge = {{
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value}};
    BaseZoneParameter<ColorVector<float>> minCellEnergy = {.baseValue = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f}};
    BaseParameter<ColorVector<float>> normalCellEnergy = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    BaseZoneParameter<ColorVector<float>> cellDeathProbability = {.baseValue = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f}};
    BaseParameter<CellDeathConsquences> cellDeathConsequences = {CellDeathConsquences_DetachedPartsDie};

    // Mutation
    BaseZoneParameter<ColorVector<float>> copyMutationNeuronData = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationCellProperties = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationGeometry = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationCustomGeometry = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationCellType = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationInsertion = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationDeletion = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationTranslation = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationDuplication = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationCellColor = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationSubgenomeColor = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorVector<float>> copyMutationGenomeColor = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorMatrix<bool>> copyMutationColorTransitions = {
        {{true, true, true, true, true, true, true},
         {true, true, true, true, true, true, true},
         {true, true, true, true, true, true, true},
         {true, true, true, true, true, true, true},
         {true, true, true, true, true, true, true},
         {true, true, true, true, true, true, true},
         {true, true, true, true, true, true, true}}};
    BaseParameter<bool> copyMutationPreventDepthIncrease = {false};
    BaseParameter<bool> copyMutationSelfReplication = {false};

    // Cell type: Attacker
    BaseZoneParameter<ColorVector<float>> attackerEnergyCost = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseZoneParameter<ColorMatrix<float>> attackerFoodChainColorMatrix = {
        .baseValue = {
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1}}};
    BaseParameter<ColorVector<float>> attackerStrength = {{0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f}};
    BaseParameter<ColorVector<float>> attackerRadius = {{1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f}};
    BaseZoneParameter<ColorMatrix<float>> attackerComplexCreatureProtection = {
        .baseValue = {
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};
    BaseParameter<bool> attackerDestroyCells = {true};

    // Cell type: Constructor
    BaseParameter<ColorVector<float>> constructorConnectingCellDistance = {{2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f}};
    BaseParameter<bool> constructorCompletenessCheck = {false};
    static float constexpr constructorAdditionalOffspringDistance = 0.8f;

    // Cell type: Defender
    BaseParameter<ColorVector<float>> defenderAntiAttackerStrength = {{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};
    BaseParameter<ColorVector<float>> defenderAntiInjectorStrength = {{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};

    // Cell type: Injector
    BaseParameter<ColorVector<float>> injectorInjectionRadius = {{3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f}};
    BaseParameter<ColorMatrix<int>> injectorInjectionTime = {{
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3}}};

    // Cell type: Muscle
    BaseParameter<ColorVector<float>> muscleEnergyCost = {{0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorVector<float>> muscleMovementAcceleration = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseParameter<ColorVector<float>> muscleCrawlingAcceleration = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseParameter<ColorVector<float>> muscleBendingAcceleration = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    static float constexpr cellTypeMuscleThreshold = 0.2f;
    static int constexpr cellTypeMuscleActivationCountdown = 10;

    // Cell type: Sensor
    BaseParameter<ColorVector<float>> sensorRadius = {{255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f}};

    // Cell type: Transmitter
    BaseParameter<ColorVector<float>> transmitterEnergyDistributionRadius = {{3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f}};
    BaseParameter<ColorVector<float>> transmitterEnergyDistributionValue = {{10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f}};
    BaseParameter<bool> transmitterEnergyDistributionSameCreature = {true};

    // Cell type: Reconnector
    BaseParameter<ColorVector<float>> reconnectorRadius = {{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}};

    // Cell type: Detonator
    BaseParameter<ColorVector<float>> detonatorRadius = {{10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f}};
    BaseParameter<ColorVector<float>> detonatorChainExplosionProbability = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};

    // Expert settings: Advanced absorption control
    ExpertToggle advancedAbsorptionControlToggle = {false};
    BaseZoneParameter<ColorVector<float>> radiationAbsorptionLowGenomeComplexityPenalty = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> radiationAbsorptionLowConnectionPenalty = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> radiationAbsorptionHighVelocityPenalty = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseZoneParameter<ColorVector<float>> radiationAbsorptionLowVelocityPenalty = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

    // Expert settings: Advanced attacker control
    ExpertToggle advancedAttackerControlToggle = {false};
    BaseParameter<ColorMatrix<float>> attackerSameMutantProtection = {{
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};
    BaseZoneParameter<ColorMatrix<float>> attackerNewComplexMutantProtection = {.baseValue = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};
    BaseParameter<ColorVector<float>> attackerSensorDetectionFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseZoneParameter<ColorVector<float>> attackerGeometryDeviationProtection = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseZoneParameter<ColorVector<float>> attackerConnectionsMismatchProtection = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    static float constexpr attackerColorInhomogeneityFactor = 1.0f;

    // Expert settings: Cell age limiter
    ExpertToggle cellAgeLimiterToggle = {false};
    BaseZoneParameter<ColorVector<float>> maxAgeForInactiveCells = {.baseValue = {  // Candidate for deletion
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value}};
    BaseParameter<ColorVector<int>> freeCellMaxAge = {{
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value}};
    BaseParameter<bool> resetCellAgeAfterActivation = {false};  // Candidate for deletion
    EnableableBaseParameter<int> maxCellAgeBalancerInterval = {.value = 10000, .enabled = false};

    // Expert settings: Cell color transition rules
    ExpertToggle colorTransitionRulesToggle = {false};
    BaseZoneParameter<ColorTransitionRules> colorTransitionRules;

    // Expert settings: Cell glow
    ExpertToggle cellGlowToggle = {false};
    BaseParameter<CellColoring> cellGlowColoring = {CellColoring_CellColor};
    BaseParameter<float> cellGlowRadius = {4.0f};
    BaseParameter<ColorVector<float>> cellGlowStrength = {{0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f}};

    // Expert settings: Customize deletion mutations setting
    ExpertToggle customizeDeletionMutationsToggle = {false};
    BaseParameter<int> cellCopyMutationDeletionMinSize = {0};

    // Expert settings: Customize neuron mutations setting
    ExpertToggle customizeNeuronMutationsToggle = {false};
    BaseParameter<float> cellCopyMutationNeuronDataWeight = {0.2f};
    BaseParameter<float> cellCopyMutationNeuronDataBias = {0.2f};
    BaseParameter<float> cellCopyMutationNeuronDataActivationFunction = {0.05f};
    BaseParameter<float> cellCopyMutationNeuronDataReinforcement = {1.05f};
    BaseParameter<float> cellCopyMutationNeuronDataDamping = {1.05f};
    BaseParameter<float> cellCopyMutationNeuronDataOffset = {0.05f};

    // Expert settings: External energy settings
    ExpertToggle externalEnergyControlToggle = {false};
    BaseParameter<float> externalEnergy = {0.0f};
    BaseParameter<ColorVector<float>> externalEnergyInflowFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> externalEnergyConditionalInflowFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<bool> externalEnergyInflowOnlyForNonSelfReplicators = {false};
    BaseParameter<ColorVector<float>> externalEnergyBackflowFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<float> externalEnergyBackflowLimit = {Infinity<float>::value};

    // Expert settings: Genome complexity measurement
    ExpertToggle genomeComplexityMeasurementToggle = {false};
    BaseParameter<ColorVector<float>> genomeComplexitySizeFactor = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseParameter<ColorVector<float>> genomeComplexityRamificationFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<int>> genomeComplexityDepthLevel = {{3, 3, 3, 3, 3, 3, 3}};

    // OLD

    // Particle sources
    RadiationSource radiationSource[MAX_RADIATION_SOURCES];

    // Spots
    SimulationParametersZone zone[MAX_ZONES];


    bool operator==(SimulationParameters const&) const = default;
};
