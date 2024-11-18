#include "LegacyAuxiliaryDataParserService.h"

#include <set>

#include "ParameterParser.h"

namespace
{
    template <typename T>
    bool equals(SimulationParameters const& parameters, ColorVector<T> SimulationParametersSpotValues::*parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if ((parameters.baseValues.*parameter)[i] != value) {
                return false;
            }
            for (int j = 0; j < parameters.numSpots; ++j) {
                if ((parameters.spots[j].values.*parameter)[i] != value) {
                    return false;
                }
            }
        }
        return true;
    }
    template <typename T>
    bool contains(SimulationParameters const& parameters, ColorVector<T> SimulationParametersSpotValues::*parameter, std::set<T> const& values)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (!values.contains((parameters.baseValues.*parameter)[i])) {
                return false;
            }
            for (int j = 0; j < parameters.numSpots; ++j) {
                if (!values.contains((parameters.spots[j].values.*parameter)[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    bool equals(ColorVector<T> const& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (parameter[i] != value) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool equals(ColorMatrix<T> const& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++i) {
                if (parameter[i][j] != value) {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename T>
    void readLegacyParameterForBase(LegacyProperty<T>& result, boost::property_tree::ptree& tree, std::string const& node)
    {
        T defaultDummy;
        result.existent = !ParameterParser::encodeDecode(tree, result.parameter, defaultDummy, node, ParserTask::Decode);
    }

    template <typename T>
    void readLegacyParameterForSpot(LegacySpotProperty<T>& result, boost::property_tree::ptree& tree, std::string const& node)
    {
        T defaultDummy;
        result.existent = !ParameterParser::encodeDecodeWithEnabled(tree, result.parameter, result.active, defaultDummy, node, ParserTask::Decode);
    }

    LegacyParametersForBase readLegacyParametersForBase(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        LegacyParametersForBase result;
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationNeuronDataProbability, tree, nodeBase + "cell.function.constructor.mutation probability.neuron data");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationPropertiesProbability, tree, nodeBase + "cell.function.constructor.mutation probability.data");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationCellFunctionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell function");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.geometry");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationCustomGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.custom geometry");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationInsertionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.insertion");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationDeletionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.deletion");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationTranslationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.translation");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationDuplicationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.duplication");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationCellColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell color");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationSubgenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.color");
        readLegacyParameterForBase(
            result.cellFunctionConstructorMutationGenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.uniform color");

        readLegacyParameterForBase(
            result.cellFunctionMuscleMovementAngleFromSensor, tree, nodeBase + "cell.function.muscle.movement angle from sensor");

        readLegacyParameterForBase(result.clusterDecay, tree, nodeBase + "cluster.decay");
        readLegacyParameterForBase(result.clusterDecayProb, tree, nodeBase + "cluster.decay probability");

        return result;
    }

    LegacyParametersForSpot readLegacyParametersForSpot(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        LegacyParametersForSpot result;
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationNeuronDataProbability, tree, nodeBase + "cell.function.constructor.mutation probability.neuron data");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationPropertiesProbability, tree, nodeBase + "cell.function.constructor.mutation probability.data ");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationCellFunctionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell function");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.geometry");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationCustomGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.custom geometry");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationInsertionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.insertion");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationDeletionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.deletion");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationTranslationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.translation");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationDuplicationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.duplication");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationCellColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell color");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationSubgenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.color");
        readLegacyParameterForSpot(
            result.cellFunctionConstructorMutationGenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.uniform color");
        return result;
    }
}

void LegacyAuxiliaryDataParserService::searchAndApplyLegacyParameters(
    std::string const& programVersion,
    boost::property_tree::ptree& tree,
    MissingFeatures const& missingFeatures,
    MissingParameters const& missingParameters,
    SimulationParameters& parameters)
{
    LegacyFeatures legacyFeatures;
    readLegacyParameterForBase(legacyFeatures.advancedMuscleControl, tree, "simulation parameters.features.additional muscle control");

    LegacyParameters legacyParameters;
    legacyParameters.base = readLegacyParametersForBase(tree, "simulation parameters.");
    for (int i = 0; i < parameters.numSpots; ++i) {
        legacyParameters.spots[i] = readLegacyParametersForSpot(tree, "simulation parameters.spots." + std::to_string(i) + ".");
    }
    updateParametersAndFeaturesForLegacyFiles(programVersion, missingFeatures, legacyFeatures, missingParameters, legacyParameters, parameters);
}

void LegacyAuxiliaryDataParserService::updateParametersAndFeaturesForLegacyFiles(
    std::string const& programVersion,
    MissingFeatures const& missingFeatures,
    LegacyFeatures const& legacyFeatures,
    MissingParameters const& missingParameters,
    LegacyParameters const& legacyParameters,
    SimulationParameters& parameters)
{
    //parameter conversion for v4.10.2 and below
    if (programVersion.empty()) {
        parameters.features.legacyModes = true;
        if (parameters.numRadiationSources > 0) {
            auto strengthRatio = 1.0f / parameters.numRadiationSources;
            for (int i = 0; i < parameters.numRadiationSources; ++i) {
                parameters.radiationSources[i].strengthRatio = strengthRatio;
            }
        }
    }

    //*******************
    //* older conversions
    //*******************
    //activation of legacyCellFunctionMuscleMovementAngleFromChannel before v4.10.0
    if (missingFeatures.legacyMode && !legacyFeatures.advancedMuscleControl.existent) {
        parameters.cellFunctionMuscleMovementTowardTargetedObject = false;
    }

    //activation of legacyCellFunctionMuscleMovementAngleFromChannel between v4.10.0 and v4.10.1
    if (legacyFeatures.advancedMuscleControl.existent && legacyParameters.base.cellFunctionMuscleMovementAngleFromSensor.existent) {
        parameters.features.legacyModes = true;
        parameters.cellFunctionMuscleMovementTowardTargetedObject =
            legacyFeatures.advancedMuscleControl.parameter && legacyParameters.base.cellFunctionMuscleMovementAngleFromSensor.parameter;
        parameters.legacyCellFunctionMuscleMovementAngleFromSensor = true;
    }

    //activation of other features
    if (missingFeatures.advancedAbsorptionControl) {
        if (!equals(parameters.radiationAbsorptionHighVelocityPenalty, 0.0f) || !equals(parameters.radiationAbsorptionLowConnectionPenalty, 0.0f)) {
            parameters.features.advancedAbsorptionControl = true;
        }
    }

    if (missingFeatures.advancedAttackerControl) {
        auto advancedAttackerControlForSpot = false;
        for (int i = 0; i < parameters.numSpots; ++i) {
            auto const& spotValues = parameters.spots[i].values;
            if (!equals(spotValues.cellFunctionAttackerGeometryDeviationExponent, 0.0f)
                || !equals(spotValues.cellFunctionAttackerConnectionsMismatchPenalty, 0.0f)) {
                advancedAttackerControlForSpot = true;
            }
        }
        if (advancedAttackerControlForSpot || !equals(parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus, 0.0f)
            || !equals(parameters.cellFunctionAttackerSameMutantPenalty, 0.0f) || !equals(parameters.cellFunctionAttackerSensorDetectionFactor, 0.0f)
            || !equals(parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent, 0.0f)
            || !equals(parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty, 0.0f)
            || !equals(parameters.cellFunctionAttackerColorInhomogeneityFactor, 1.0f) || !equals(parameters.cellFunctionAttackerEnergyDistributionRadius, 3.6f)
            || !equals(parameters.cellFunctionAttackerEnergyDistributionValue, 10.0f)) {
            parameters.features.advancedAttackerControl = true;
        }
    }

    if (missingFeatures.externalEnergyControl) {
        if (parameters.externalEnergy != 0.0f || !equals(parameters.externalEnergyInflowFactor, 0.0f)
            || !equals(parameters.externalEnergyConditionalInflowFactor, 0.0f) || !equals(parameters.externalEnergyBackflowFactor, 0.0f)) {
            parameters.features.externalEnergyControl = true;
        }
    }

    if (missingFeatures.cellColorTransitionRules) {
        if (!contains(parameters, &SimulationParametersSpotValues::cellColorTransitionDuration, {0, Infinity<int>::value})) {
            parameters.features.cellColorTransitionRules = true;
        }
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (parameters.baseValues.cellColorTransitionTargetColor[i] != i) {
                parameters.features.cellColorTransitionRules = true;
                break;
            }
            for (int j = 0; j < parameters.numSpots; ++j) {
                if (parameters.spots[j].values.cellColorTransitionTargetColor[i] != i) {
                    parameters.features.cellColorTransitionRules = true;
                    break;
                }
            }
        }
    }

    if (missingFeatures.cellAgeLimiter && parameters.cellMaxAgeBalancer) {
        parameters.features.cellAgeLimiter = true;
    }

    //activation of externalEnergyBackflowFactor
    if (missingParameters.externalEnergyBackflowFactor) {
        if (!equals(parameters.externalEnergyConditionalInflowFactor, 0.0f)) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                parameters.externalEnergyBackflowFactor[i] = parameters.externalEnergyConditionalInflowFactor[i] / 5;
            }
        }
    }

    //conversion of mutation rates to genome copy mutations
    if (missingParameters.copyMutations) {
        auto setParametersForBase = [](SimulationParametersSpotValues& target, LegacyParametersForBase const& source) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                target.cellCopyMutationNeuronData[i] = source.cellFunctionConstructorMutationNeuronDataProbability.parameter[i] * 250;
                target.cellCopyMutationCellProperties[i] = source.cellFunctionConstructorMutationPropertiesProbability.parameter[i] * 250;
                target.cellCopyMutationCellFunction[i] = source.cellFunctionConstructorMutationCellFunctionProbability.parameter[i] * 250;
                target.cellCopyMutationGeometry[i] = source.cellFunctionConstructorMutationGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationCustomGeometry[i] = source.cellFunctionConstructorMutationCustomGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationInsertion[i] = source.cellFunctionConstructorMutationInsertionProbability.parameter[i] * 250;
                target.cellCopyMutationDeletion[i] = source.cellFunctionConstructorMutationDeletionProbability.parameter[i] * 250;
                target.cellCopyMutationCellColor[i] = source.cellFunctionConstructorMutationCellColorProbability.parameter[i] * 250;
                target.cellCopyMutationTranslation[i] = source.cellFunctionConstructorMutationTranslationProbability.parameter[i] * 5000;
                target.cellCopyMutationDuplication[i] = source.cellFunctionConstructorMutationDuplicationProbability.parameter[i] * 5000;
                target.cellCopyMutationSubgenomeColor[i] = source.cellFunctionConstructorMutationSubgenomeColorProbability.parameter[i] * 5000;
                target.cellCopyMutationGenomeColor[i] = source.cellFunctionConstructorMutationGenomeColorProbability.parameter[i] * 5000;
            }
        };
        auto setParametersForSpot = [](SimulationParametersSpotValues& target, LegacyParametersForSpot const& source) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                target.cellCopyMutationNeuronData[i] = source.cellFunctionConstructorMutationNeuronDataProbability.parameter[i] * 250;
                target.cellCopyMutationCellProperties[i] = source.cellFunctionConstructorMutationPropertiesProbability.parameter[i] * 250;
                target.cellCopyMutationCellFunction[i] = source.cellFunctionConstructorMutationCellFunctionProbability.parameter[i] * 250;
                target.cellCopyMutationGeometry[i] = source.cellFunctionConstructorMutationGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationCustomGeometry[i] = source.cellFunctionConstructorMutationCustomGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationInsertion[i] = source.cellFunctionConstructorMutationInsertionProbability.parameter[i] * 250;
                target.cellCopyMutationDeletion[i] = source.cellFunctionConstructorMutationDeletionProbability.parameter[i] * 250;
                target.cellCopyMutationCellColor[i] = source.cellFunctionConstructorMutationCellColorProbability.parameter[i] * 250;
                target.cellCopyMutationTranslation[i] = source.cellFunctionConstructorMutationTranslationProbability.parameter[i] * 5000;
                target.cellCopyMutationDuplication[i] = source.cellFunctionConstructorMutationDuplicationProbability.parameter[i] * 5000;
                target.cellCopyMutationSubgenomeColor[i] = source.cellFunctionConstructorMutationSubgenomeColorProbability.parameter[i] * 5000;
                target.cellCopyMutationGenomeColor[i] = source.cellFunctionConstructorMutationGenomeColorProbability.parameter[i] * 5000;
            }
        };

        setParametersForBase(parameters.baseValues, legacyParameters.base);
        for (int i = 0; i < MAX_SPOTS; ++i) {
            setParametersForSpot(parameters.spots->values, legacyParameters.spots[i]);
        }
    }

    //conversion of cluster decay and probabilities
    if (missingParameters.cellDeathConsequences) {
        parameters.cellDeathConsequences = legacyParameters.base.clusterDecay.parameter ? CellDeathConsquences_CreatureDies : CellDeathConsquences_None;
        if (parameters.cellDeathConsequences == CellDeathConsquences_None) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                parameters.baseValues.cellDeathProbability[i] = 0.01f;
            }
        } else {
            for (int i = 0; i < MAX_COLORS; ++i) {
                parameters.baseValues.cellDeathProbability[i] = legacyParameters.base.clusterDecayProb.parameter[i];
            }
        }
    }
}
