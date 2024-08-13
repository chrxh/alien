#pragma once
#include <optional>

#include "SimulationParameters.h"

struct MissingParameters
{
    bool externalEnergyBackflowFactor = false;
    bool copyMutations = false;
};

template<typename T>
struct LegacyParameter
{
    bool active = false;
    T parameter;
};

struct LegacyParametersForSpot
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
struct LegacyParameters
{
    LegacyParametersForSpot base;
    LegacyParametersForSpot spots[MAX_SPOTS];
};

class LegacySimulationParametersService
{
public:
    static void activateFeaturesForLegacyFiles(Features const& missingFeatures, SimulationParameters& parameters);
    static void activateParametersForLegacyFiles(MissingParameters const& missingParameters, LegacyParameters const& legacyParameters, SimulationParameters& parameters);
};
