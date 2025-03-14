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
    //general
    Char64 projectName = "<unnamed>";

    //feature list
    Features features;

    //particle sources
    int numRadiationSources = 0;
    RadiationSource radiationSource[MAX_RADIATION_SOURCES];
    bool baseStrengthRatioPinned = false;

    float externalEnergy = 0.0f;
    ColorVector<float> externalEnergyInflowFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> externalEnergyConditionalInflowFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> externalEnergyBackflowFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool externalEnergyInflowOnlyForNonSelfReplicators = false;
    float externalEnergyBackflowLimit = Infinity<float>::value;

    //spots
    int numZones = 0;
    SimulationParametersZone zone[MAX_ZONES];

    //rendering
    uint32_t backgroundColor = 0x1b0000;
    bool borderlessRendering = false;
    bool markReferenceDomain = true;
    bool gridLines = false;
    CellColoring cellColoring = CellColoring_CellColor;
    CellColoring cellGlowColoring = CellColoring_CellColor;
    float cellGlowRadius = 4.0f;
    float cellGlowStrength = 0.1f;
    CellType highlightedCellType = CellType_Constructor;
    float zoomLevelNeuronalActivity = 2.0f;
    bool attackVisualization = false;
    bool muscleMovementVisualization = false;
    float cellRadius = 0.25f;
    bool showRadiationSources = true;

    //all other parameters
    SimulationParametersZoneValues baseValues;

    float timestepSize = 1.0f;
    Motion motionData;

    float innerFriction = 0.3f;
    float cellMaxVelocity = 2.0f;
    ColorVector<float> cellMaxBindingDistance = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};

    ColorVector<float> cellNormalEnergy = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    float cellMinDistance = 0.3f;
    float cellMaxForceDecayProb = 0.2f;

    ColorVector<float> genomeComplexitySizeFactor = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> genomeComplexityRamificationFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> genomeComplexityNeuronFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<int> genomeComplexityDepthLevel = {3, 3, 3, 3, 3, 3, 3};

    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;
    ColorVector<int> radiationMinCellAge = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> radiationAbsorptionHighVelocityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationAbsorptionLowConnectionPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> highRadiationFactor = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> highRadiationMinCellEnergy = {500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f};
    CellDeathConsquences cellDeathConsequences = CellDeathConsquences_DetachedPartsDie;
    ColorVector<int> cellMaxAge = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    bool cellInactiveMaxAgeActivated = false;
    bool cellEmergentMaxAgeActivated = false;
    ColorVector<int> cellEmergentMaxAge = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};

    bool cellMaxAgeBalancer = false;
    int cellMaxAgeBalancerInterval = 10000;
    bool cellResetAgeAfterActivation = false;

    bool particleTransformationAllowed = false;
    ColorVector<float> particleSplitEnergy = {
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value};

    ColorMatrix<bool> cellCopyMutationColorTransitions = {
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true},
        {true, true, true, true, true, true, true}};
    bool cellCopyMutationPreventDepthIncrease = false;
    bool cellCopyMutationSelfReplication = false;

    // Expert settings: Customize neuron mutations setting
    float cellCopyMutationNeuronDataWeight = 0.2f;
    float cellCopyMutationNeuronDataBias = 0.2f;
    float cellCopyMutationNeuronDataActivationFunction = 0.05f;
    float cellCopyMutationNeuronDataReinforcement = 1.05f;
    float cellCopyMutationNeuronDataDamping = 1.05f;
    float cellCopyMutationNeuronDataOffset = 0.05f;

    // Expert settings: Customize deletion mutations setting
    int cellCopyMutationDeletionMinSize = 0;

    // Cell type: Injector
    ColorVector<float> cellTypeInjectorRadius = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    ColorMatrix<int> cellTypeInjectorDurationColorMatrix = {
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3}};

    // Cell type: Attacker
    ColorVector<float> cellTypeAttackerRadius = {1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f};
    ColorVector<float> cellTypeAttackerStrength = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    ColorVector<float> cellTypeAttackerEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    ColorVector<float> cellTypeAttackerEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    ColorVector<float> cellTypeAttackerColorInhomogeneityFactor = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorMatrix<float> cellTypeAttackerSameMutantPenalty = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    ColorVector<float> cellTypeAttackerSensorDetectionFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool cellTypeAttackerDestroyCells = false;

    // Cell type: Constructor
    ColorVector<float> cellTypeConstructorConnectingCellMaxDistance = {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f};
    float cellTypeConstructorAdditionalOffspringDistance = 0.8f;
    bool cellTypeConstructorCheckCompletenessForSelfReplication = false;

    // Cell type: Defender
    ColorVector<float> cellTypeDefenderAgainstAttackerStrength = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};
    ColorVector<float> cellTypeDefenderAgainstInjectorStrength = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};

    // Cell type: Transmitter
    bool cellTypeTransmitterEnergyDistributionSameCreature = true;
    ColorVector<float> cellTypeTransmitterEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    ColorVector<float> cellTypeTransmitterEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

    // Cell type: Muscle
    ColorVector<float> cellTypeMuscleBendingAcceleration = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> cellTypeMuscleCrawlingAcceleration = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> cellTypeMuscleMovementAcceleration = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> cellTypeMuscleEnergyCost = {0, 0, 0, 0, 0, 0, 0};
    static float constexpr cellTypeMuscleThreshold = 0.2f;
    static int constexpr cellTypeMuscleActivationCountdown = 10;

    // Cell type: Sensor
    ColorVector<float> cellTypeSensorRange = {255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f};

    // Cell type: Reconnector
    ColorVector<float> cellTypeReconnectorRadius = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    // Cell type: Detonator
    ColorVector<float> cellTypeDetonatorRadius = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    ColorVector<float> cellTypeDetonatorChainExplosionProbability = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    bool operator==(SimulationParameters const&) const = default;
};
