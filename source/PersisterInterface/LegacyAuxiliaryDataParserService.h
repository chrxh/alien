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
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationNeuronDataProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationPropertiesProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationCellFunctionProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationGeometryProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationCustomGeometryProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationInsertionProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationDeletionProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationTranslationProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationDuplicationProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationCellColorProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationSubgenomeColorProbability;
    LegacyProperty<ColorVector<float>> cellFunctionConstructorMutationGenomeColorProbability;
    LegacyProperty<bool> cellFunctionMuscleMovementAngleFromSensor;
    LegacyProperty<bool> clusterDecay;
    LegacyProperty<ColorVector<float>> clusterDecayProb;
};

struct LegacyParametersForSpot
{
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationNeuronDataProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationPropertiesProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationCellFunctionProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationGeometryProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationCustomGeometryProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationInsertionProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationDeletionProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationTranslationProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationDuplicationProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationCellColorProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationSubgenomeColorProbability;
    LegacySpotProperty<ColorVector<float>> cellFunctionConstructorMutationGenomeColorProbability;
};

struct LegacyParameters
{
    LegacyParametersForBase base;
    LegacyParametersForSpot spots[MAX_SPOTS];
};

class LegacyAuxiliaryDataParserService
{
    MAKE_SINGLETON(LegacyAuxiliaryDataParserService);

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
