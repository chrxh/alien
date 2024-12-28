#pragma once

#include <cstdint>
#include <cstring>

#include "CellFunctionConstants.h"
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
    CellFunction highlightedCellFunction = CellFunction_Constructor;
    float zoomLevelNeuronalActivity = 2.0f;
    bool attackVisualization = false;
    bool muscleMovementVisualization = false;
    float cellRadius = 0.25f;
    bool showRadiationSources = true;

    //all other parameters
    SimulationParametersZoneValues baseValues;

    float timestepSize = 1.0f;
    MotionType motionType = MotionType_Fluid;
    MotionData motionData = {FluidMotion()};

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
    bool particleTransformationRandomCellFunction = false;
    int particleTransformationMaxGenomeSize = 300;
    ColorVector<float> particleSplitEnergy = {
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value};

    ColorVector<float> cellFunctionConstructorConnectingCellMaxDistance = {1.8f, 1.8f, 1.8f, 1.8f, 1.8f, 1.8f, 1.8f};
    ColorVector<float> cellFunctionConstructorSignalThreshold = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    bool cellFunctionConstructorCheckCompletenessForSelfReplication = false;

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

    // customize neuron mutations setting
    float cellCopyMutationNeuronDataWeight = 0.2f;
    float cellCopyMutationNeuronDataBias = 0.2f;
    float cellCopyMutationNeuronDataActivationFunction = 0.05f;
    float cellCopyMutationNeuronDataReinforcement = 1.05f;
    float cellCopyMutationNeuronDataDamping = 1.05f;
    float cellCopyMutationNeuronDataOffset = 0.05f;

    // customize deletion mutations setting
    int cellCopyMutationDeletionMinSize = 0;

    ColorVector<float> cellFunctionInjectorRadius = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    ColorMatrix<int> cellFunctionInjectorDurationColorMatrix = {
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3}};
    float cellFunctionInjectorSignalThreshold = 0.1f;

    ColorVector<float> cellFunctionAttackerRadius = {1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f};
    ColorVector<float> cellFunctionAttackerStrength = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    ColorVector<float> cellFunctionAttackerEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    ColorVector<float> cellFunctionAttackerEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    ColorVector<float> cellFunctionAttackerColorInhomogeneityFactor = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorMatrix<float> cellFunctionAttackerSameMutantPenalty = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    ColorVector<float> cellFunctionAttackerSensorDetectionFactor = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float cellFunctionAttackerSignalThreshold = 0.1f;
    bool cellFunctionAttackerDestroyCells = false;

    ColorVector<float> cellFunctionDefenderAgainstAttackerStrength = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};
    ColorVector<float> cellFunctionDefenderAgainstInjectorStrength = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};

    bool cellFunctionTransmitterEnergyDistributionSameCreature = true;
    ColorVector<float> cellFunctionTransmitterEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    ColorVector<float> cellFunctionTransmitterEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

    ColorVector<float> cellFunctionMuscleContractionExpansionDelta = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    ColorVector<float> cellFunctionMuscleMovementAcceleration = {0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f};
    ColorVector<float> cellFunctionMuscleBendingAngle = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    ColorVector<float> cellFunctionMuscleBendingAcceleration = {0.15f, 0.15f, 0.15f, 0.15f, 0.15f, 0.15f, 0.15f};
    float cellFunctionMuscleBendingAccelerationThreshold = 0.1f;
    bool cellFunctionMuscleMovementTowardTargetedObject = true;
    ColorVector<float> cellFunctionMuscleEnergyCost = {0, 0, 0, 0, 0, 0, 0};

    ColorVector<float> cellFunctionSensorRange = {255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f};
    float cellFunctionSensorSignalThreshold = 0.1f;

    ColorVector<float> cellFunctionReconnectorRadius = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float cellFunctionReconnectorSignalThreshold = 0.1f;

    ColorVector<float> cellFunctionDetonatorRadius = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    ColorVector<float> cellFunctionDetonatorChainExplosionProbability = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float cellFunctionDetonatorSignalThreshold = 0.1f;

    bool legacyCellFunctionMuscleMovementAngleFromSensor = false;

    bool operator==(SimulationParameters const& other) const;
    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
