#pragma once

#include <optional>

#include <boost/property_tree/ptree_fwd.hpp>

#include "Base/Singleton.h"
#include "EngineInterface/SimulationParameters.h"

struct MissingFeatures
{
    bool advancedAbsorptionControl = false;
    bool advancedAttackerControl = false;
    bool externalEnergyControl = false;
    bool cellColorTransitionRules = false;
    bool cellAgeLimiter = false;
    bool legacyMode = false;
};

struct MissingParameters
{
    bool externalEnergyBackflowFactor = false;
    bool copyMutations = false;
    bool cellDeathConsequences = false;
};

template <typename T>
struct LegacyProperty
{
    bool existent = false;
    T parameter;
};

template <typename T>
struct LegacySpotProperty
{
    bool existent = false;
    bool active = false;
    T parameter;
};

struct LegacyFeatures
{
    LegacyProperty<bool> advancedMuscleControl;
};

struct LegacyParametersForBase
{
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationNeuronDataProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationPropertiesProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationCellTypeProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationGeometryProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationCustomGeometryProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationInsertionProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationDeletionProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationTranslationProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationDuplicationProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationCellColorProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationSubgenomeColorProbability;
    LegacyProperty<ColorVector<float>> cellTypeConstructorMutationGenomeColorProbability;
    LegacyProperty<bool> cellTypeMuscleMovementAngleFromSensor;
    LegacyProperty<bool> clusterDecay;
    LegacyProperty<ColorVector<float>> clusterDecayProb;
};

struct LegacyParametersForSpot
{
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationNeuronDataProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationPropertiesProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationCellTypeProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationGeometryProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationCustomGeometryProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationInsertionProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationDeletionProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationTranslationProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationDuplicationProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationCellColorProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationSubgenomeColorProbability;
    LegacySpotProperty<ColorVector<float>> cellTypeConstructorMutationGenomeColorProbability;
};

struct LegacyParameters
{
    LegacyParametersForBase base;
    LegacyParametersForSpot spots[MAX_ZONES];
};

class LegacySettingsParserService
{
    MAKE_SINGLETON(LegacySettingsParserService);

public:
    //Note: missingFeatures and missingParameters are deprecated, use programVersion instead
    void searchAndApplyLegacyParameters(
        std::string const& programVersion,
        boost::property_tree::ptree& tree,
        MissingFeatures const& missingFeatures,
        MissingParameters const& missingParameters,
        SimulationParameters& parameters);

private:
    void updateParametersAndFeaturesForLegacyFiles(
        std::string const& programVersion,
        MissingFeatures const& missingFeatures,
        LegacyFeatures const& legacyFeatures,
        MissingParameters const& missingParameters,
        LegacyParameters const& legacyParameters,
        SimulationParameters& parameters);
};
