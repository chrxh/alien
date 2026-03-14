#pragma once

#include <cstdint>
#include <cstring>

#include <Base/MathTypes.h>

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
    int layerOrderNumbers[MAX_LAYERS] = {};
    int sourceOrderNumbers[MAX_SOURCES] = {};

    // General
    BaseParameter<Char64> projectName = {"<unnamed simulation>"};
    LayerParameter<Char64> layerName = {{"<unnamed>"}};
    SourceParameter<Char64> sourceName = {{"<unnamed>"}};
    LayerParameter<float> layerOpacity = {{1.0f}};
    PinnableSourceParameter<float> sourceRelativeStrength = {{{.value = 0.0f, .pinned = false}}};

    // Visualization
    BaseLayerParameter<FloatColorRGB> backgroundColor = {.baseValue = {0.0f, 0.0f, 0.106f}};
    BaseParameter<CellColoring> cellColoring = {CellColoring_CellColor};
    BaseParameter<CellType> highlightedCellType = {CellType_Base};
    BaseParameter<float> glow = {0.3f};
    BaseParameter<bool> borderlessRendering = {false};
    BaseParameter<bool> gridLines = {false};
    BaseParameter<bool> markReferenceDomain = {true};
    SourceParameter<bool> sourceShowRadiationCenter = {true};

    // Location
    LayerParameter<RealVector2D> layerPosition;
    LayerParameter<RealVector2D> layerVelocity;
    LayerParameter<LayerShapeType> layerShape = {{LayerShapeType_Circular}};
    LayerParameter<float> layerCoreRadius = {{100.0f}};                 
    LayerParameter<RealVector2D> layerCoreRect = {{{100.0f, 100.0f}}};  
    LayerParameter<float> layerFadeoutRadius = {{100.0f}};
    SourceParameter<SourceShapeType> sourceShapeType = {{SourceShapeType_Circular}};
    SourceParameter<float> sourceCircularRadius = {{1.0f}};                    
    SourceParameter<RealVector2D> sourceRectangularRect = {{{30.0f, 60.0f}}};  
    SourceParameter<RealVector2D> sourcePosition;
    SourceParameter<RealVector2D> sourceVelocity;

    // Force field
    EnableableLayerParameter<ForceField> layerForceFieldType = {.layerValues = {{ForceField_None, false}}};
    LayerParameter<Orientation> layerRadialForceFieldOrientation = {{Orientation_Clockwise}};  
    LayerParameter<float> layerRadialForceFieldStrength = {{0.001f}};                          
    LayerParameter<float> layerRadialForceFieldDriftAngle = {{0.0f}};                          
    LayerParameter<float> layerCentralForceFieldStrength = {{0.05f}};                          
    LayerParameter<float> layerLinearForceFieldAngle = {{0}};
    LayerParameter<float> layerLinearForceFieldStrength = {{0.0001f}};
    LayerParameter<float> layerPerlinNoiseForceFieldStrength = {{0.001f}};      
    LayerParameter<float> layerPerlinNoiseForceFieldSpatialSize = {{20.0f}};    
    LayerParameter<float> layerPerlinNoiseForceFieldTemporalSize = {{1000.0f}}; 

    // Physics: Motion
    BaseParameter<float> timestepSize = {1.0f};
    BaseParameter<float> smoothingLength = {0.8f};    // for MotionType_Fluid
    BaseParameter<float> viscosityStrength = {0.1f};  // for MotionType_Fluid
    BaseParameter<float> pressureStrength = {0.1f};   // for MotionType_Fluid
    BaseLayerParameter<float> friction = {.baseValue = 0.001f};
    BaseParameter<float> innerFriction = {0.6f};
    BaseLayerParameter<float> rigidity = {.baseValue = 0.0f};

    // Physics: Thresholds
    BaseParameter<float> maxVelocity = {2.0f};
    BaseLayerParameter<ColorVector<float>> maxForce = {.baseValue = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f}};
    BaseParameter<float> minObjectDistance = {0.3f};
    BaseParameter<ColorVector<float>> maxBindingDistance = {{3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f}};
    BaseLayerParameter<float> objectFusionVelocity = {.baseValue = 0.1f};
    static float constexpr maxAcceleration = 0.4f;
    static float constexpr maxForceDecayProbability = 0.2f;

    // Radiation
    PinBaseParameter relativeStrengthBasePin = {false};
    LayerParameter<bool> disableRadiationSources = {{false}};
    BaseLayerParameter<ColorVector<float>> radiationAbsorption = {.baseValue = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseLayerParameter<ColorVector<float>> radiationType1_strength = {
        .baseValue = {0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f}};
    BaseParameter<ColorVector<int>> radiationType1_minimumAge = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorVector<float>> radiationType2_strength = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorVector<float>> radiationType2_energyThreshold = {500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f};
    BaseParameter<ColorVector<float>> particleSplitEnergy = {{50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f}};
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
         Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value}};
    BaseLayerParameter<ColorVector<float>> minCellEnergy = {.baseValue = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f}};
    BaseParameter<ColorVector<float>> normalCellEnergy = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    BaseLayerParameter<ColorVector<float>> cellDeathProbability = {
        .baseValue = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f}};
    BaseParameter<CellDeathConsequences> cellDeathConsequences = {CellDeathConsequences_DetachedPartsDie};

    // Cell constructor
    BaseParameter<ColorVector<float>> constructorConnectingCellDistance = {{3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f}};

    // Meta mutations
    BaseParameter<float> metaMutationNeuronsSigma = {0};
    BaseParameter<float> metaMutationConnectionsSigma = {0};
    BaseParameter<float> metaMutationLineagesSigma = {0};

    // Cell type: Attacker
    BaseLayerParameter<ColorVector<float>> attackerEnergyCost = {.baseValue = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    BaseLayerParameter<ColorMatrix<float>> attackerFoodChainColorMatrix = {
        .baseValue = {
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}};
    BaseParameter<ColorVector<float>> attackerRelatedLineageProtection = {{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};
    BaseParameter<ColorVector<float>> attackerStrength = {{0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f}};
    BaseParameter<ColorVector<float>> attackerRadius = {{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}};
    static float constexpr attackerMaxRawEnergyThreshold = 2.0f;
    static float constexpr attackerCreatureSensorRange = 5.0f;

    // Cell type: Digestor
    BaseParameter<ColorVector<float>> maxRawEnergyConductivity = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    BaseParameter<ColorVector<float>> maxRawEnergyConversion = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    static float constexpr maxRawEnergyThresholdForConduction = 100.0f;

    // Cell type: Defender
    BaseParameter<ColorVector<float>> defenderAntiAttackerStrength = {{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};
    BaseParameter<ColorVector<float>> defenderAntiInjectorStrength = {{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};

    // Cell type: Injector
    BaseParameter<ColorVector<float>> injectorEnergyCost = {{50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f}};
    BaseParameter<ColorVector<float>> injectorRadius = {{3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f}};

    // Cell type: Muscle
    BaseParameter<ColorVector<float>> muscleEnergyCost = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    BaseParameter<ColorVector<float>> muscleMovementAcceleration = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseParameter<ColorVector<float>> muscleCrawlingAcceleration = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseParameter<ColorVector<float>> muscleBendingAcceleration = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    static int constexpr muscleActivationCountdown = 10;

    // Cell type: Sensor
    BaseParameter<ColorVector<float>> sensorRadius = {{255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f}};

    // Cell type: Depot
    static float constexpr depotEnergyTransferUnit = 2.0f;
    static float constexpr depotStorageLimit = 500.0f;

    // Cell type: Reconnector
    BaseParameter<ColorVector<float>> reconnectorRadius = {{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}};

    // Cell type: Detonator
    BaseParameter<ColorVector<float>> detonatorRadius = {{10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f}};
    BaseParameter<ColorVector<float>> detonatorChainExplosionProbability = {{0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f}};

    // Expert settings: Advanced absorption control
    ExpertToggle advancedAbsorptionControlToggle = {false};
    BaseLayerParameter<ColorVector<float>> radiationAbsorptionLowNumCellsPenalty = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> radiationAbsorptionLowConnectionPenalty = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<ColorVector<float>> radiationAbsorptionHighVelocityPenalty = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseLayerParameter<ColorVector<float>> radiationAbsorptionLowVelocityPenalty = {.baseValue = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

    // Expert settings: Cell age limiter
    ExpertToggle cellAgeLimiterToggle = {false};
    BaseParameter<ColorVector<int>> freeCellMaxAge = {
        {Infinity<int>::value,
         Infinity<int>::value,
         Infinity<int>::value,
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
    BaseLayerParameter<ColorVector<ColorTransitionRule>> colorTransitionRules;

    // Expert settings: Customize deletion mutations setting
    ExpertToggle customizeDeletionMutationsToggle = {false};
    BaseParameter<int> cellCopyMutationDeletionMinSize = {0};

    // Expert settings: External energy settings
    ExpertToggle externalEnergyControlToggle = {false};
    BaseParameter<float> externalEnergy = {0.0f};
    BaseParameter<ColorVector<float>> externalEnergyInflowFactor = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    BaseParameter<ColorVector<float>> externalEnergyConditionalInflowFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<bool> externalEnergyInflowOnlyForNonSelfReplicators = {false};
    BaseParameter<ColorVector<float>> externalEnergyBackflowFactor = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    BaseParameter<float> externalEnergyBackflowLimit = {Infinity<float>::value};

    bool operator==(SimulationParameters const&) const = default;

    static ParametersSpec const& getSpec();
};
