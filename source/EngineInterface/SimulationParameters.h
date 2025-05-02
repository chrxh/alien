#pragma once

#include <cstdint>
#include <cstring>

#include "Base/Vector2D.h"

#include "CellTypeConstants.h"
#include "SimulationParametersTypes.h"

struct ParametersSpec;

/**
 * NOTE: header is also included in kernel code
 */
struct SimulationParameters
{
    int numLayers = 0;
    int numSources = 0;
    int layerLocationIndex[MAX_LAYERS] = {};
    int sourceLocationIndex[MAX_SOURCES] = {};

    // General
    BaseParameter<Char64> projectName = {"<unnamed>"};
    LayerParameter<Char64> layerName = {{"<unnamed>"}};
    SourceParameter<Char64> sourceName = {{"<unnamed>"}};

    // Visualization
    BaseLayerParameter<FloatColorRGB> backgroundColor = {.baseValue = {0.0f, 0.0f, 0.106f}};
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
    LayerParameter<RealVector2D> layerPosition;
    LayerParameter<RealVector2D> layerVelocity;
    LayerParameter<LayerShapeType> layerShape = {{LayerShapeType_Circular}};
    LayerParameter<float> layerCoreRadius = {{100.0f}};                 // for LayerShapeType_Circular
    LayerParameter<RealVector2D> layerCoreRect = {{{100.0f, 100.0f}}};  // for LayerShapeType_Rectangular
    LayerParameter<float> layerFadeoutRadius = {{100.0f}};
    SourceParameter<SourceShapeType> sourceShapeType = {{SourceShapeType_Circular}};
    SourceParameter<float> sourceCircularRadius = {{1.0f}};                    // for SourceShapeType_Circular
    SourceParameter<RealVector2D> sourceRectangularRect = {{{30.0f, 60.0f}}};  // for SourceShapeType_Rectangular
    SourceParameter<RealVector2D> sourcePosition;
    SourceParameter<RealVector2D> sourceVelocity;

    // Force field
    LayerParameter<ForceField> layerForceFieldType = {{ForceField_None}};
    LayerParameter<Orientation> layerRadialForceFieldOrientation = {{Orientation_Clockwise}};  // for ForceField_Radial
    LayerParameter<float> layerRadialForceFieldStrength = {{0.001f}};                          // for ForceField_Radial
    LayerParameter<float> layerRadialForceFieldDriftAngle = {{0.0f}};                          // for ForceField_Radial
    LayerParameter<float> layerCentralForceFieldStrength = {{0.05f}};                          // for ForceField_Central
    LayerParameter<float> layerLinearForceFieldAngle = {{0}};
    LayerParameter<float> layerLinearForceFieldStrength = {{0.01f}};

    // Numerics
    BaseParameter<float> timestepSize = {1.0f};

    // Physics: Motion
    BaseParameter<MotionType> motionType = {MotionType_Fluid};
    BaseParameter<float> smoothingLength = {0.8f};       // for MotionType_Fluid
    BaseParameter<float> viscosityStrength = {0.1f};     // for MotionType_Fluid
    BaseParameter<float> pressureStrength = {0.1f};      // for MotionType_Fluid
    BaseParameter<float> maxCollisionDistance = {1.3f};  // for MotionType_Collision
    BaseParameter<float> repulsionStrength = {0.08f};    // for MotionType_Collision
    BaseLayerParameter<float> friction = {.baseValue = 0.01f};
    BaseParameter<float> innerFriction = {0.3f};
    BaseLayerParameter<float> rigidity = {.baseValue = 0.0f};

    // Physics: Thresholds
    BaseParameter<float> maxVelocity = {2.0f};
    BaseLayerParameter<ColorVector<float>> maxForce = {.baseValue = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f}};
    BaseParameter<float> minCellDistance = {0.3f};
    static float constexpr maxForceDecayProbability = 0.2f;

    // Physics: Binding
    BaseParameter<ColorVector<float>> maxBindingDistance = {{3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f}};
    BaseLayerParameter<float> cellFusionVelocity = {.baseValue = 2.0f};
    BaseLayerParameter<float> cellMaxBindingEnergy = {.baseValue = Infinity<float>::value};

    // Radiation
    PinBaseParameter relativeStrengthBasePin = {false};
    PinnableSourceParameter<float> sourceRelativeStrength = {{{.value = 0.0f, .pinned = false}}};
    LayerParameter<bool> disableRadiationSources = {{false}};
    BaseLayerParameter<ColorVector<float>> radiationAbsorption = {.baseValue = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseLayerParameter<ColorVector<float>> radiationType1_strength = {.baseValue = {0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f}};
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
    EnableableSourceParameter<float> sourceRadiationAngle = {{{.value = 0.0f, .enabled = false}}};
    static float constexpr radiationProbability = 0.03f;
    static float constexpr radiationVelocityMultiplier = 1.0f;
    static float constexpr radiationVelocityPerturbation = 0.5f;

    // Cell life cycle
    BaseParameter<ColorVector<int>> maxCellAge = {
        {Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value}};
    BaseLayerParameter<ColorVector<float>> minCellEnergy = {.baseValue = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f}};
    BaseParameter<ColorVector<float>> normalCellEnergy = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    BaseLayerParameter<ColorVector<float>> cellDeathProbability = {.baseValue = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f}};
    BaseParameter<CellDeathConsquences> cellDeathConsequences = {CellDeathConsquences_DetachedPartsDie};

    // Mutation
    BaseLayerParameter<ColorVector<float>> copyMutationNeuronData = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationCellProperties = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationGeometry = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationCustomGeometry = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationCellType = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationInsertion = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationDeletion = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationTranslation = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationDuplication = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationCellColor = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationSubgenomeColor = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorVector<float>> copyMutationGenomeColor = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
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
    BaseLayerParameter<ColorVector<float>> attackerEnergyCost = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorMatrix<float>> attackerFoodChainColorMatrix = {
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
    BaseLayerParameter<ColorMatrix<float>> attackerComplexCreatureProtection = {
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
    BaseParameter<ColorMatrix<int>> injectorInjectionTime = {
        {{3, 3, 3, 3, 3, 3, 3},
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
    BaseLayerParameter<ColorVector<float>> radiationAbsorptionLowGenomeComplexityPenalty = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> radiationAbsorptionLowConnectionPenalty = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> radiationAbsorptionHighVelocityPenalty = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseLayerParameter<ColorVector<float>> radiationAbsorptionLowVelocityPenalty = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

    // Expert settings: Advanced attacker control
    ExpertToggle advancedAttackerControlToggle = {false};
    BaseParameter<ColorMatrix<float>> attackerSameMutantProtection = {
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};
    BaseLayerParameter<ColorMatrix<float>> attackerNewComplexMutantProtection = {
        .baseValue = {
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};
    BaseParameter<ColorVector<float>> attackerSensorDetectionFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseLayerParameter<ColorVector<float>> attackerGeometryDeviationProtection = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseLayerParameter<ColorVector<float>> attackerConnectionsMismatchProtection = {.baseValue = {0, 0, 0, 0, 0, 0, 0}};
    static float constexpr attackerColorInhomogeneityFactor = 1.0f;

    // Expert settings: Cell age limiter
    ExpertToggle cellAgeLimiterToggle = {false};
    BaseLayerParameter<ColorVector<float>> maxAgeForInactiveCells = {
        .baseValue = {// Candidate for deletion
                      Infinity<float>::value,
                      Infinity<float>::value,
                      Infinity<float>::value,
                      Infinity<float>::value,
                      Infinity<float>::value,
                      Infinity<float>::value,
                      Infinity<float>::value}};
    BaseParameter<ColorVector<int>> freeCellMaxAge = {
        {Infinity<int>::value,
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
    BaseLayerParameter<ColorTransitionRules> colorTransitionRules;

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

    bool operator==(SimulationParameters const&) const = default;

    static ParametersSpec const& getSpec();
};

