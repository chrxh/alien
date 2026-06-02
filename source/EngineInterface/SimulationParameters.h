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
    BaseParameter<ColorVector<FloatColorRGB>> customizationColors = {getDefaultCustomizationColorVector()};
    BaseParameter<ColorVector<CellColoring>> objectColoring = {ColorVector<CellColoring>::uniform(CellColoring_CellColor)};
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
    LayerParameter<float> layerPerlinNoiseForceFieldTemporalSize = {{10000.0f}};

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
    BaseLayerParameter<ColorVector<float>> maxForce = {.baseValue = ColorVector<float>::uniform(0.8f)};
    BaseParameter<float> minObjectDistance = {0.3f};
    BaseParameter<ColorVector<float>> maxBindingDistance = {ColorVector<float>::uniform(3.6f)};
    BaseLayerParameter<float> objectFusionVelocity = {.baseValue = 0.1f};
    static float constexpr maxAcceleration = 0.4f;
    static float constexpr maxForceDecayProbability = 0.2f;

    // Radiation
    PinBaseParameter relativeStrengthBasePin = {false};
    LayerParameter<bool> disableRadiationSources = {{false}};
    BaseLayerParameter<ColorVector<float>> radiationAbsorption = {.baseValue = ColorVector<float>::uniform(1.0f)};
    BaseLayerParameter<ColorVector<float>> radiationType1_strength = {.baseValue = ColorVector<float>::uniform(0.0f)};
    BaseParameter<ColorVector<int>> radiationType1_minimumAge = {ColorVector<int>::uniform(0)};
    BaseParameter<ColorVector<float>> radiationType2_strength = {ColorVector<float>::uniform(0.0f)};
    BaseParameter<ColorVector<float>> radiationType2_energyThreshold = {ColorVector<float>::uniform(500.0f)};
    BaseParameter<ColorVector<float>> particleSplitEnergy = {ColorVector<float>::uniform(50.0f)};
    BaseParameter<bool> particleTransformationAllowed = {false};
    EnableableSourceParameter<float> sourceRadiationAngle = {{{.value = 0.0f, .enabled = false}}};
    static float constexpr radiationProbability = 0.03f;
    static float constexpr radiationVelocityMultiplier = 1.0f;
    static float constexpr radiationVelocityPerturbation = 0.5f;

    // Cell life cycle
    BaseParameter<ColorVector<int>> maxCellAge = {ColorVector<int>::uniform(Infinity<int>::value)};
    BaseLayerParameter<ColorVector<float>> minCellEnergy = {.baseValue = ColorVector<float>::uniform(50.0f)};
    BaseParameter<ColorVector<float>> normalCellEnergy = {ColorVector<float>::uniform(100.0f)};
    BaseLayerParameter<ColorVector<float>> cellDeathProbability = {.baseValue = ColorVector<float>::uniform(0.001f)};

    // Cell constructor
    BaseParameter<ColorVector<float>> constructorConnectingCellDistance = {ColorVector<float>::uniform(3.5f)};

    // Meta mutations
    BaseParameter<float> metaMutationNeuronsSigma = {0};
    BaseParameter<float> metaMutationConnectionsSigma = {0};
    BaseParameter<float> accumulatedMutationsForNewLineage = {10.0f};

    // Cell type: Attacker
    BaseLayerParameter<ColorVector<float>> attackerEnergyCost = {.baseValue = ColorVector<float>::uniform(0.0f)};
    BaseLayerParameter<ColorMatrix<float>> attackerFoodChainColorMatrix = {.baseValue = ColorMatrix<float>::uniform(1.0f)};
    BaseParameter<ColorVector<float>> attackerRelatedLineageProtection = {ColorVector<float>::uniform(0.0f)};
    BaseParameter<float> attackerStrength = {0.05f};
    BaseParameter<ColorVector<float>> attackerRadius = {ColorVector<float>::uniform(2.0f)};
    static float constexpr attackerMaxRawEnergyThreshold = 2.0f;
    static float constexpr attackerCreatureSensorRange = 5.0f;

    // Cell type: Digestor
    BaseParameter<ColorVector<float>> maxRawEnergyConductivity = {ColorVector<float>::uniform(3.0f)};
    BaseParameter<ColorVector<float>> maxRawEnergyConversion = {ColorVector<float>::uniform(0.1f)};
    static float constexpr maxRawEnergyThresholdForConduction = 100.0f;

    // Cell type: Defender
    BaseParameter<ColorVector<float>> defenderAntiAttackerStrength = {ColorVector<float>::uniform(0.5f)};
    BaseParameter<ColorVector<float>> defenderAntiInjectorStrength = {ColorVector<float>::uniform(0.5f)};

    // Cell type: Injector
    BaseParameter<ColorVector<float>> injectorEnergyCost = {ColorVector<float>::uniform(50.0f)};
    BaseParameter<ColorVector<float>> injectorRadius = {ColorVector<float>::uniform(3.0f)};

    // Cell type: Muscle
    BaseParameter<ColorVector<float>> muscleEnergyCost = {ColorVector<float>::uniform(0.0f)};
    BaseParameter<ColorVector<float>> muscleMovementAcceleration = {ColorVector<float>::uniform(1.0f)};
    BaseParameter<ColorVector<float>> muscleCrawlingAcceleration = {ColorVector<float>::uniform(1.0f)};
    BaseParameter<ColorVector<float>> muscleBendingAcceleration = {ColorVector<float>::uniform(1.0f)};
    static int constexpr muscleActivationCountdown = 10;

    // Cell type: Sensor
    BaseParameter<ColorVector<float>> sensorRadius = {ColorVector<float>::uniform(255.0f)};

    // Cell type: Depot
    static float constexpr depotEnergyTransferUnit = 2.0f;
    static float constexpr depotStorageLimit = 500.0f;

    // Cell type: Reconnector
    BaseParameter<ColorVector<float>> reconnectorRadius = {ColorVector<float>::uniform(2.0f)};

    // Cell type: Detonator
    BaseParameter<ColorVector<float>> detonatorRadius = {ColorVector<float>::uniform(10.0f)};
    BaseParameter<ColorVector<float>> detonatorChainExplosionProbability = {ColorVector<float>::uniform(0.2f)};

    // Expert settings: Advanced absorption control
    ExpertToggle advancedAbsorptionControlToggle = {false};
    BaseLayerParameter<ColorVector<float>> radiationAbsorptionLowNumCellsPenalty = {.baseValue = ColorVector<float>::uniform(0.0f)};
    BaseParameter<ColorVector<float>> radiationAbsorptionLowConnectionPenalty = {ColorVector<float>::uniform(0.0f)};
    BaseParameter<ColorVector<float>> radiationAbsorptionHighVelocityPenalty = {ColorVector<float>::uniform(0.0f)};
    BaseLayerParameter<ColorVector<float>> radiationAbsorptionLowVelocityPenalty = {.baseValue = ColorVector<float>::uniform(0.0f)};

    // Expert settings: Cell age limiter
    ExpertToggle cellAgeLimiterToggle = {false};
    BaseParameter<ColorVector<int>> freeCellMaxAge = {ColorVector<int>::uniform(Infinity<int>::value)};
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
    BaseParameter<ColorVector<float>> externalEnergyInflowFactor = {ColorVector<float>::uniform(1.0f)};
    BaseParameter<ColorVector<float>> externalEnergyInflowThresholdFactor = {ColorVector<float>::uniform(0.0f)};
    BaseParameter<bool> externalEnergyInflowOnlyForFirstOffspring = {false};
    BaseParameter<ColorVector<float>> externalEnergyBackflowFactor = {ColorVector<float>::uniform(0.0f)};
    BaseParameter<float> externalEnergyBackflowLimit = {Infinity<float>::value};

    bool operator==(SimulationParameters const&) const = default;

    static ParametersSpec const& getSpec();
};
