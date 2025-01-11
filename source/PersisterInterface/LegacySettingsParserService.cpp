#include "LegacySettingsParserService.h"

#include <set>

#include "Base/StringHelper.h"
#include "Base/VersionParserService.h"

#include "ParameterParser.h"

namespace
{
    template <typename T>
    bool equals(SimulationParameters const& parameters, ColorVector<T> SimulationParametersZoneValues::*parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if ((parameters.baseValues.*parameter)[i] != value) {
                return false;
            }
            for (int j = 0; j < parameters.numZones; ++j) {
                if ((parameters.zone[j].values.*parameter)[i] != value) {
                    return false;
                }
            }
        }
        return true;
    }
    template <typename T>
    bool contains(SimulationParameters const& parameters, ColorVector<T> SimulationParametersZoneValues::*parameter, std::set<T> const& values)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (!values.contains((parameters.baseValues.*parameter)[i])) {
                return false;
            }
            for (int j = 0; j < parameters.numZones; ++j) {
                if (!values.contains((parameters.zone[j].values.*parameter)[i])) {
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
            result.cellTypeConstructorMutationNeuronDataProbability, tree, nodeBase + "cell.function.constructor.mutation probability.neuron data");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationPropertiesProbability, tree, nodeBase + "cell.function.constructor.mutation probability.data");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationCellTypeProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell function");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.geometry");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationCustomGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.custom geometry");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationInsertionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.insertion");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationDeletionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.deletion");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationTranslationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.translation");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationDuplicationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.duplication");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationCellColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell color");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationSubgenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.color");
        readLegacyParameterForBase(
            result.cellTypeConstructorMutationGenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.uniform color");

        readLegacyParameterForBase(
            result.cellTypeMuscleMovementAngleFromSensor, tree, nodeBase + "cell.function.muscle.movement angle from sensor");

        readLegacyParameterForBase(result.clusterDecay, tree, nodeBase + "cluster.decay");
        readLegacyParameterForBase(result.clusterDecayProb, tree, nodeBase + "cluster.decay probability");

        return result;
    }

    LegacyParametersForSpot readLegacyParametersForSpot(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        LegacyParametersForSpot result;
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationNeuronDataProbability, tree, nodeBase + "cell.function.constructor.mutation probability.neuron data");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationPropertiesProbability, tree, nodeBase + "cell.function.constructor.mutation probability.data ");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationCellTypeProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell function");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.geometry");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationCustomGeometryProbability, tree, nodeBase + "cell.function.constructor.mutation probability.custom geometry");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationInsertionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.insertion");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationDeletionProbability, tree, nodeBase + "cell.function.constructor.mutation probability.deletion");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationTranslationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.translation");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationDuplicationProbability, tree, nodeBase + "cell.function.constructor.mutation probability.duplication");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationCellColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.cell color");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationSubgenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.color");
        readLegacyParameterForSpot(
            result.cellTypeConstructorMutationGenomeColorProbability, tree, nodeBase + "cell.function.constructor.mutation probability.uniform color");
        return result;
    }
}

void LegacySettingsParserService::searchAndApplyLegacyParameters(
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
    for (int i = 0; i < parameters.numZones; ++i) {
        legacyParameters.spots[i] = readLegacyParametersForSpot(tree, "simulation parameters.spots." + std::to_string(i) + ".");
    }
    updateParametersAndFeaturesForLegacyFiles(programVersion, missingFeatures, legacyFeatures, missingParameters, legacyParameters, parameters);
}

void LegacySettingsParserService::updateParametersAndFeaturesForLegacyFiles(
    std::string const& programVersion,
    MissingFeatures const& missingFeatures,
    LegacyFeatures const& legacyFeatures,
    MissingParameters const& missingParameters,
    LegacyParameters const& legacyParameters,
    SimulationParameters& parameters)
{
    //parameter conversion for v4.10.x and below
    if (programVersion.empty()) {
        parameters.features.legacyModes = true;
        if (parameters.numRadiationSources > 0) {
            auto strengthRatio = 1.0f / parameters.numRadiationSources;
            for (int i = 0; i < parameters.numRadiationSources; ++i) {
                parameters.radiationSource[i].strength = strengthRatio;
            }
            parameters.baseStrengthRatioPinned = true;
        }
    }

    //parameter conversion for v4.11.x and below
    auto versionParts = VersionParserService::get().getVersionParts(programVersion);
    if (versionParts.major <= 4 && versionParts.minor <= 11) {
        int locationPosition = 0;
        if (parameters.numZones > 0) {
            for (int i = 0; i < parameters.numZones; ++i) {
                parameters.zone[i].locationIndex = ++locationPosition;
                StringHelper::copy(parameters.zone[i].name, sizeof(parameters.zone[i].name), "Zone " + std::to_string(i + 1));
            }
        }
        if (parameters.numRadiationSources > 0) {
            for (int i = 0; i < parameters.numRadiationSources; ++i) {
                parameters.radiationSource[i].locationIndex = ++locationPosition;
                StringHelper::copy(parameters.radiationSource[i].name, sizeof(parameters.zone[i].name), "Radiation " + std::to_string(i + 1));
            }
        }
    }

    //*******************
    //* older conversions
    //*******************
    //activation of legacyCellTypeMuscleMovementAngleFromChannel before v4.10.0
    if (missingFeatures.legacyMode && !legacyFeatures.advancedMuscleControl.existent) {
        parameters.cellTypeMuscleMovementTowardTargetedObject = false;
    }

    //activation of legacyCellTypeMuscleMovementAngleFromChannel between v4.10.0 and v4.10.1
    if (legacyFeatures.advancedMuscleControl.existent && legacyParameters.base.cellTypeMuscleMovementAngleFromSensor.existent) {
        parameters.features.legacyModes = true;
        parameters.cellTypeMuscleMovementTowardTargetedObject =
            legacyFeatures.advancedMuscleControl.parameter && legacyParameters.base.cellTypeMuscleMovementAngleFromSensor.parameter;
        parameters.legacyCellTypeMuscleMovementAngleFromSensor = true;
    }

    //activation of other features
    if (missingFeatures.advancedAbsorptionControl) {
        if (!equals(parameters.radiationAbsorptionHighVelocityPenalty, 0.0f) || !equals(parameters.radiationAbsorptionLowConnectionPenalty, 0.0f)) {
            parameters.features.advancedAbsorptionControl = true;
        }
    }

    if (missingFeatures.advancedAttackerControl) {
        auto advancedAttackerControlForSpot = false;
        for (int i = 0; i < parameters.numZones; ++i) {
            auto const& spotValues = parameters.zone[i].values;
            if (!equals(spotValues.cellTypeAttackerGeometryDeviationExponent, 0.0f)
                || !equals(spotValues.cellTypeAttackerConnectionsMismatchPenalty, 0.0f)) {
                advancedAttackerControlForSpot = true;
            }
        }
        if (advancedAttackerControlForSpot
            || !equals(parameters.cellTypeAttackerSameMutantPenalty, 0.0f) || !equals(parameters.cellTypeAttackerSensorDetectionFactor, 0.0f)
            || !equals(parameters.baseValues.cellTypeAttackerGeometryDeviationExponent, 0.0f)
            || !equals(parameters.baseValues.cellTypeAttackerConnectionsMismatchPenalty, 0.0f)
            || !equals(parameters.cellTypeAttackerColorInhomogeneityFactor, 1.0f) || !equals(parameters.cellTypeAttackerEnergyDistributionRadius, 3.6f)
            || !equals(parameters.cellTypeAttackerEnergyDistributionValue, 10.0f)) {
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
        if (!contains(parameters, &SimulationParametersZoneValues::cellColorTransitionDuration, {0, Infinity<int>::value})) {
            parameters.features.cellColorTransitionRules = true;
        }
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (parameters.baseValues.cellColorTransitionTargetColor[i] != i) {
                parameters.features.cellColorTransitionRules = true;
                break;
            }
            for (int j = 0; j < parameters.numZones; ++j) {
                if (parameters.zone[j].values.cellColorTransitionTargetColor[i] != i) {
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
        auto setParametersForBase = [](SimulationParametersZoneValues& target, LegacyParametersForBase const& source) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                target.cellCopyMutationNeuronData[i] = source.cellTypeConstructorMutationNeuronDataProbability.parameter[i] * 250;
                target.cellCopyMutationCellProperties[i] = source.cellTypeConstructorMutationPropertiesProbability.parameter[i] * 250;
                target.cellCopyMutationCellType[i] = source.cellTypeConstructorMutationCellTypeProbability.parameter[i] * 250;
                target.cellCopyMutationGeometry[i] = source.cellTypeConstructorMutationGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationCustomGeometry[i] = source.cellTypeConstructorMutationCustomGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationInsertion[i] = source.cellTypeConstructorMutationInsertionProbability.parameter[i] * 250;
                target.cellCopyMutationDeletion[i] = source.cellTypeConstructorMutationDeletionProbability.parameter[i] * 250;
                target.cellCopyMutationCellColor[i] = source.cellTypeConstructorMutationCellColorProbability.parameter[i] * 250;
                target.cellCopyMutationTranslation[i] = source.cellTypeConstructorMutationTranslationProbability.parameter[i] * 5000;
                target.cellCopyMutationDuplication[i] = source.cellTypeConstructorMutationDuplicationProbability.parameter[i] * 5000;
                target.cellCopyMutationSubgenomeColor[i] = source.cellTypeConstructorMutationSubgenomeColorProbability.parameter[i] * 5000;
                target.cellCopyMutationGenomeColor[i] = source.cellTypeConstructorMutationGenomeColorProbability.parameter[i] * 5000;
            }
        };
        auto setParametersForSpot = [](SimulationParametersZoneValues& target, LegacyParametersForSpot const& source) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                target.cellCopyMutationNeuronData[i] = source.cellTypeConstructorMutationNeuronDataProbability.parameter[i] * 250;
                target.cellCopyMutationCellProperties[i] = source.cellTypeConstructorMutationPropertiesProbability.parameter[i] * 250;
                target.cellCopyMutationCellType[i] = source.cellTypeConstructorMutationCellTypeProbability.parameter[i] * 250;
                target.cellCopyMutationGeometry[i] = source.cellTypeConstructorMutationGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationCustomGeometry[i] = source.cellTypeConstructorMutationCustomGeometryProbability.parameter[i] * 250;
                target.cellCopyMutationInsertion[i] = source.cellTypeConstructorMutationInsertionProbability.parameter[i] * 250;
                target.cellCopyMutationDeletion[i] = source.cellTypeConstructorMutationDeletionProbability.parameter[i] * 250;
                target.cellCopyMutationCellColor[i] = source.cellTypeConstructorMutationCellColorProbability.parameter[i] * 250;
                target.cellCopyMutationTranslation[i] = source.cellTypeConstructorMutationTranslationProbability.parameter[i] * 5000;
                target.cellCopyMutationDuplication[i] = source.cellTypeConstructorMutationDuplicationProbability.parameter[i] * 5000;
                target.cellCopyMutationSubgenomeColor[i] = source.cellTypeConstructorMutationSubgenomeColorProbability.parameter[i] * 5000;
                target.cellCopyMutationGenomeColor[i] = source.cellTypeConstructorMutationGenomeColorProbability.parameter[i] * 5000;
            }
        };

        setParametersForBase(parameters.baseValues, legacyParameters.base);
        for (int i = 0; i < MAX_ZONES; ++i) {
            setParametersForSpot(parameters.zone->values, legacyParameters.spots[i]);
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
