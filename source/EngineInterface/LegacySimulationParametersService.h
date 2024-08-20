#pragma once
#include <optional>

#include "SimulationParameters.h"

struct MissingParameters
{
    bool externalEnergyBackflowFactor = false;
    bool copyMutations = false;
};

template<typename T>
struct LegacySpotParameter
{
    bool active = false;
    T parameter;
};

struct LegacyParametersForBase
{
    ColorVector<float> cellFunctionConstructorMutationNeuronDataProbability;
    ColorVector<float> cellFunctionConstructorMutationPropertiesProbability;
    ColorVector<float> cellFunctionConstructorMutationCellFunctionProbability;
    ColorVector<float> cellFunctionConstructorMutationGeometryProbability;
    ColorVector<float> cellFunctionConstructorMutationCustomGeometryProbability;
    ColorVector<float> cellFunctionConstructorMutationInsertionProbability;
    ColorVector<float> cellFunctionConstructorMutationDeletionProbability;
    ColorVector<float> cellFunctionConstructorMutationTranslationProbability;
    ColorVector<float> cellFunctionConstructorMutationDuplicationProbability;
    ColorVector<float> cellFunctionConstructorMutationCellColorProbability;
    ColorVector<float> cellFunctionConstructorMutationSubgenomeColorProbability;
    ColorVector<float> cellFunctionConstructorMutationGenomeColorProbability;
};

struct LegacyParametersForSpot
{
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationNeuronDataProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationPropertiesProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationCellFunctionProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationGeometryProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationCustomGeometryProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationInsertionProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationDeletionProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationTranslationProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationDuplicationProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationCellColorProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationSubgenomeColorProbability;
    LegacySpotParameter<ColorVector<float>> cellFunctionConstructorMutationGenomeColorProbability;
};

struct LegacyParameters
{
    LegacyParametersForBase base;
    LegacyParametersForSpot spots[MAX_SPOTS];
};

class LegacySimulationParametersService
{
public:
    static void activateFeaturesForLegacyFiles(Features const& missingFeatures, SimulationParameters& parameters);
    static void activateParametersForLegacyFiles(MissingParameters const& missingParameters, LegacyParameters const& legacyParameters, SimulationParameters& parameters);
};
