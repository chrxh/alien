#include "AuxiliaryDataParserService.h"

#include "GeneralSettings.h"
#include "Settings.h"
#include "LegacySimulationParametersService.h"

namespace
{
    //return true if value does not exist in tree
    template <typename T>
    bool encodeDecodeProperty(boost::property_tree::ptree& tree, T& parameter, T const& defaultValue, std::string const& node, ParserTask task)
    {
        return JsonParser::encodeDecode(tree, parameter, defaultValue, node, task);
    }

    template <>
    bool encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        ColorVector<float>& parameter,
        ColorVector<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result |= encodeDecodeProperty(tree, parameter[i], defaultValue[i], node + "[" + std::to_string(i) + "]", task);
        }
        return result;
    }

    template <>
    bool encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        ColorVector<int>& parameter,
        ColorVector<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result |= encodeDecodeProperty(tree, parameter[i], defaultValue[i], node + "[" + std::to_string(i) + "]", task);
        }
        return result;
    }

    template <>
    bool encodeDecodeProperty<ColorMatrix<float>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<float>& parameter,
        ColorMatrix<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result |=
                    encodeDecodeProperty(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
        return result;
    }

    template <>
    bool encodeDecodeProperty<ColorMatrix<int>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<int>& parameter,
        ColorMatrix<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result |=
                    encodeDecodeProperty(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
        return result;
    }

    template <>
    bool encodeDecodeProperty<ColorMatrix<bool>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<bool>& parameter,
        ColorMatrix<bool> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result |=
                    encodeDecodeProperty(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
        return result;
    }

    template <>
    bool encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        std::chrono::milliseconds& parameter,
        std::chrono::milliseconds const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        if (ParserTask::Encode == task) {
            auto parameterAsString = std::to_string(parameter.count());
            return encodeDecodeProperty(tree, parameterAsString, std::string(), node, task);
        } else {
            std::string parameterAsString;
            auto defaultAsString = std::to_string(defaultValue.count());
            auto result = encodeDecodeProperty(tree, parameterAsString, defaultAsString, node, task);
            parameter = std::chrono::milliseconds(std::stoi(parameterAsString));
            return result;
        }
    }

    template <typename T>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        T& parameter,
        bool& isActivated,
        T const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node + ".value", task);
    }

    template <>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        ColorVector<float>& parameter,
        bool& isActivated,
        ColorVector<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node, task);
    }

    template <>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        ColorVector<int>& parameter,
        bool& isActivated,
        ColorVector<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node, task);
    }

    template <typename T>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        ColorMatrix<T>& parameter,
        bool& isActivated,
        ColorMatrix<bool> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node, task);
    }

    void readLegacyParameterForBase(ColorVector<float>& result, boost::property_tree::ptree& tree, std::string const& node)
    {
        ColorVector<float> defaultDummy;
        encodeDecodeProperty(tree, result, defaultDummy, node, ParserTask::Decode);
    }

    void readLegacyParameterForSpot(LegacySpotParameter<ColorVector<float>>& result, boost::property_tree::ptree& tree, std::string const& node)
    {
        ColorVector<float> defaultDummy;
        encodeDecodeSpotProperty(tree, result.parameter, result.active, defaultDummy, node, ParserTask::Decode);
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

    void searchAndApplyLegacyParameters(
        boost::property_tree::ptree& tree,
        Features const& missingFeatures,
        MissingParameters const& missingParameters,
        SimulationParameters& parameters)
    {
        LegacySimulationParametersService::activateFeaturesForLegacyFiles(missingFeatures, parameters);

        LegacyParameters legacyParameters;
        legacyParameters.base = readLegacyParametersForBase(tree, "simulation parameters.");
        for (int i = 0; i < parameters.numSpots; ++i) {
            legacyParameters.spots[i] = readLegacyParametersForSpot(tree, "simulation parameters.spots." + std::to_string(i) + ".");
        }
        LegacySimulationParametersService::activateParametersForLegacyFiles(missingParameters, legacyParameters, parameters);
    }

    void encodeDecode(
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters,
        MissingParameters& missingParameters,
        Features& missingFeatures,
        ParserTask parserTask)
    {
        SimulationParameters defaultParameters;

        encodeDecodeProperty(tree, parameters.backgroundColor, defaultParameters.backgroundColor, "simulation parameters.background color", parserTask);
        encodeDecodeProperty(tree, parameters.cellColoring, defaultParameters.cellColoring, "simulation parameters.cell colorization", parserTask);
        encodeDecodeProperty(
            tree, parameters.cellGlowColoring, defaultParameters.cellGlowColoring, "simulation parameters.cell glow.coloring", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellGlowRadius,
            defaultParameters.cellGlowRadius,
            "simulation parameters.cell glow.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellGlowStrength,
            defaultParameters.cellGlowStrength,
            "simulation parameters.cell glow.strength",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.highlightedCellFunction,
            defaultParameters.highlightedCellFunction,
            "simulation parameters.highlighted cell function",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.zoomLevelNeuronalActivity,
            defaultParameters.zoomLevelNeuronalActivity,
            "simulation parameters.zoom level.neural activity",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.borderlessRendering, defaultParameters.borderlessRendering, "simulation parameters.borderless rendering", parserTask);
        encodeDecodeProperty(
            tree, parameters.markReferenceDomain, defaultParameters.markReferenceDomain, "simulation parameters.mark reference domain", parserTask);
        encodeDecodeProperty(tree, parameters.gridLines, defaultParameters.gridLines, "simulation parameters.grid lines", parserTask);
        encodeDecodeProperty(
            tree, parameters.attackVisualization, defaultParameters.attackVisualization, "simulation parameters.attack visualization", parserTask);
        encodeDecodeProperty(tree, parameters.cellRadius, defaultParameters.cellRadius, "simulation parameters.cek", parserTask);

        encodeDecodeProperty(tree, parameters.timestepSize, defaultParameters.timestepSize, "simulation parameters.time step size", parserTask);

        encodeDecodeProperty(tree, parameters.motionType, defaultParameters.motionType, "simulation parameters.motion.type", parserTask);
        if (parameters.motionType == MotionType_Fluid) {
            encodeDecodeProperty(
                tree,
                parameters.motionData.fluidMotion.smoothingLength,
                defaultParameters.motionData.fluidMotion.smoothingLength,
                "simulation parameters.fluid.smoothing length",
                parserTask);
            encodeDecodeProperty(
                tree,
                parameters.motionData.fluidMotion.pressureStrength,
                defaultParameters.motionData.fluidMotion.pressureStrength,
                "simulation parameters.fluid.pressure strength",
                parserTask);
            encodeDecodeProperty(
                tree,
                parameters.motionData.fluidMotion.viscosityStrength,
                defaultParameters.motionData.fluidMotion.viscosityStrength,
                "simulation parameters.fluid.viscosity strength",
                parserTask);
        } else {
            encodeDecodeProperty(
                tree,
                parameters.motionData.collisionMotion.cellMaxCollisionDistance,
                defaultParameters.motionData.collisionMotion.cellMaxCollisionDistance,
                "simulation parameters.motion.collision.max distance",
                parserTask);
            encodeDecodeProperty(
                tree,
                parameters.motionData.collisionMotion.cellRepulsionStrength,
                defaultParameters.motionData.collisionMotion.cellRepulsionStrength,
                "simulation parameters.motion.collision.repulsion strength",
                parserTask);
        }

        encodeDecodeProperty(tree, parameters.baseValues.friction, defaultParameters.baseValues.friction, "simulation parameters.friction", parserTask);
        encodeDecodeProperty(tree, parameters.baseValues.rigidity, defaultParameters.baseValues.rigidity, "simulation parameters.rigidity", parserTask);
        encodeDecodeProperty(tree, parameters.cellMaxVelocity, defaultParameters.cellMaxVelocity, "simulation parameters.cell.max velocity", parserTask);
        encodeDecodeProperty(
            tree, parameters.cellMaxBindingDistance, defaultParameters.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
        encodeDecodeProperty(tree, parameters.cellNormalEnergy, defaultParameters.cellNormalEnergy, "simulation parameters.cell.normal energy", parserTask);

        encodeDecodeProperty(tree, parameters.cellMinDistance, defaultParameters.cellMinDistance, "simulation parameters.cell.min distance", parserTask);
        encodeDecodeProperty(
            tree, parameters.baseValues.cellMaxForce, defaultParameters.baseValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellMaxForceDecayProb,
            defaultParameters.cellMaxForceDecayProb,
            "simulation parameters.cell.max force decay probability",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellNumExecutionOrderNumbers,
            defaultParameters.cellNumExecutionOrderNumbers,
            "simulation parameters.cell.max execution order number",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.baseValues.cellMinEnergy, defaultParameters.baseValues.cellMinEnergy, "simulation parameters.cell.min energy", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFusionVelocity,
            defaultParameters.baseValues.cellFusionVelocity,
            "simulation parameters.cell.fusion velocity",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellMaxBindingEnergy,
            parameters.baseValues.cellMaxBindingEnergy,
            "simulation parameters.cell.max binding energy",
            parserTask);
        encodeDecodeProperty(tree, parameters.cellMaxAge, defaultParameters.cellMaxAge, "simulation parameters.cell.max age", parserTask);
        encodeDecodeProperty(
            tree, parameters.cellMaxAgeBalancer, defaultParameters.cellMaxAgeBalancer, "simulation parameters.cell.max age.balance.enabled", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellMaxAgeBalancerInterval,
            defaultParameters.cellMaxAgeBalancerInterval,
            "simulation parameters.cell.max age.balance.interval",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellInactiveMaxAgeActivated,
            defaultParameters.cellInactiveMaxAgeActivated,
            "simulation parameters.cell.inactive max age activated",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellInactiveMaxAge,
            defaultParameters.baseValues.cellInactiveMaxAge,
            "simulation parameters.cell.inactive max age",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellEmergentMaxAgeActivated,
            defaultParameters.cellEmergentMaxAgeActivated,
            "simulation parameters.cell.nutrient max age activated",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.cellEmergentMaxAge, defaultParameters.cellEmergentMaxAge, "simulation parameters.cell.nutrient max age", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellResetAgeAfterActivation,
            defaultParameters.cellResetAgeAfterActivation,
            "simulation parameters.cell.reset age after activation",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellColorTransitionDuration,
            defaultParameters.baseValues.cellColorTransitionDuration,
            "simulation parameters.cell.color transition rules.duration",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellColorTransitionTargetColor,
            defaultParameters.baseValues.cellColorTransitionTargetColor,
            "simulation parameters.cell.color transition rules.target color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.genomeComplexityRamificationFactor,
            defaultParameters.genomeComplexityRamificationFactor,
            "simulation parameters.genome complexity.genome complexity ramification factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.genomeComplexitySizeFactor,
            defaultParameters.genomeComplexitySizeFactor,
            "simulation parameters.genome complexity.genome complexity size factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationCellAgeStrength,
            defaultParameters.baseValues.radiationCellAgeStrength,
            "simulation parameters.radiation.factor",
            parserTask);
        encodeDecodeProperty(tree, parameters.radiationProb, defaultParameters.radiationProb, "simulation parameters.radiation.probability", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationVelocityMultiplier,
            defaultParameters.radiationVelocityMultiplier,
            "simulation parameters.radiation.velocity multiplier",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationVelocityPerturbation,
            defaultParameters.radiationVelocityPerturbation,
            "simulation parameters.radiation.velocity perturbation",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationDisableSources,
            defaultParameters.baseValues.radiationDisableSources,
            "simulation parameters.radiation.disable sources",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationAbsorption,
            defaultParameters.baseValues.radiationAbsorption,
            "simulation parameters.radiation.absorption",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationAbsorptionHighVelocityPenalty,
            defaultParameters.radiationAbsorptionHighVelocityPenalty,
            "simulation parameters.radiation.absorption velocity penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationAbsorptionLowVelocityPenalty,
            defaultParameters.baseValues.radiationAbsorptionLowVelocityPenalty,
            "simulation parameters.radiation.absorption low velocity penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationAbsorptionLowConnectionPenalty,
            defaultParameters.radiationAbsorptionLowConnectionPenalty,
            "simulation parameters.radiation.absorption low connection penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty,
            defaultParameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty,
            "simulation parameters.radiation.absorption low genome complexity penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.highRadiationMinCellEnergy,
            defaultParameters.highRadiationMinCellEnergy,
            "simulation parameters.high radiation.min cell energy",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.highRadiationFactor, defaultParameters.highRadiationFactor, "simulation parameters.high radiation.factor", parserTask);
        encodeDecodeProperty(
            tree, parameters.radiationMinCellAge, defaultParameters.radiationMinCellAge, "simulation parameters.radiation.min cell age", parserTask);

        encodeDecodeProperty(
            tree, parameters.externalEnergy, defaultParameters.externalEnergy, "simulation parameters.cell.function.constructor.external energy", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.externalEnergyInflowFactor,
            defaultParameters.externalEnergyInflowFactor,
            "simulation parameters.cell.function.constructor.external energy supply rate",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.externalEnergyConditionalInflowFactor,
            defaultParameters.externalEnergyConditionalInflowFactor,
            "simulation parameters.cell.function.constructor.pump energy factor",
            parserTask);
        missingParameters.externalEnergyBackflowFactor = encodeDecodeProperty(
            tree,
            parameters.externalEnergyBackflowFactor,
            defaultParameters.externalEnergyBackflowFactor,
            "simulation parameters.cell.function.constructor.external energy backflow",
            parserTask);

        encodeDecodeProperty(tree, parameters.clusterDecay, defaultParameters.clusterDecay, "simulation parameters.cluster.decay", parserTask);
        encodeDecodeProperty(
            tree, parameters.clusterDecayProb, defaultParameters.clusterDecayProb, "simulation parameters.cluster.decay probability", parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorOffspringDistance,
            defaultParameters.cellFunctionConstructorOffspringDistance,
            "simulation parameters.cell.function.constructor.offspring distance",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorConnectingCellMaxDistance,
            defaultParameters.cellFunctionConstructorConnectingCellMaxDistance,
            "simulation parameters.cell.function.constructor.connecting cell max distance",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorActivityThreshold,
            defaultParameters.cellFunctionConstructorActivityThreshold,
            "simulation parameters.cell.function.constructor.activity threshold",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorCheckCompletenessForSelfReplication,
            defaultParameters.cellFunctionConstructorCheckCompletenessForSelfReplication,
            "simulation parameters.cell.function.constructor.completeness check for self-replication",
            parserTask);

        missingParameters.copyMutations = encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationNeuronData,
            defaultParameters.baseValues.cellCopyMutationNeuronData,
            "simulation parameters.cell.copy mutation.neuron data",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationCellProperties,
            defaultParameters.baseValues.cellCopyMutationCellProperties,
            "simulation parameters.cell.copy mutation.cell properties",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationGeometry,
            defaultParameters.baseValues.cellCopyMutationGeometry,
            "simulation parameters.cell.copy mutation.geometry",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationCustomGeometry,
            defaultParameters.baseValues.cellCopyMutationCustomGeometry,
            "simulation parameters.cell.copy mutation.custom geometry",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationCellFunction,
            defaultParameters.baseValues.cellCopyMutationCellFunction,
            "simulation parameters.cell.copy mutation.cell function",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationInsertion,
            defaultParameters.baseValues.cellCopyMutationInsertion,
            "simulation parameters.cell.copy mutation.insertion",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationDeletion,
            defaultParameters.baseValues.cellCopyMutationDeletion,
            "simulation parameters.cell.copy mutation.deletion",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationTranslation,
            defaultParameters.baseValues.cellCopyMutationTranslation,
            "simulation parameters.cell.copy mutation.translation",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationDuplication,
            defaultParameters.baseValues.cellCopyMutationDuplication,
            "simulation parameters.cell.copy mutation.duplication",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationCellColor,
            defaultParameters.baseValues.cellCopyMutationCellColor,
            "simulation parameters.cell.copy mutation.cell color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationSubgenomeColor,
            defaultParameters.baseValues.cellCopyMutationSubgenomeColor,
            "simulation parameters.cell.copy mutation.subgenome color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellCopyMutationGenomeColor,
            defaultParameters.baseValues.cellCopyMutationGenomeColor,
            "simulation parameters.cell.copy mutation.genome color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorMutationColorTransitions,
            defaultParameters.cellFunctionConstructorMutationColorTransitions,
            "simulation parameters.cell.copy mutation.color transition",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorMutationSelfReplication,
            defaultParameters.cellFunctionConstructorMutationSelfReplication,
            "simulation parameters.cell.copy mutation.self replication flag",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorMutationPreventDepthIncrease,
            defaultParameters.cellFunctionConstructorMutationPreventDepthIncrease,
            "simulation parameters.cell.copy mutation.prevent depth increase",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionInjectorRadius,
            defaultParameters.cellFunctionInjectorRadius,
            "simulation parameters.cell.function.injector.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionInjectorDurationColorMatrix,
            defaultParameters.cellFunctionInjectorDurationColorMatrix,
            "simulation parameters.cell.function.injector.duration",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerRadius,
            defaultParameters.cellFunctionAttackerRadius,
            "simulation parameters.cell.function.attacker.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerStrength,
            defaultParameters.cellFunctionAttackerStrength,
            "simulation parameters.cell.function.attacker.strength",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerEnergyDistributionRadius,
            defaultParameters.cellFunctionAttackerEnergyDistributionRadius,
            "simulation parameters.cell.function.attacker.energy distribution radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerEnergyDistributionValue,
            defaultParameters.cellFunctionAttackerEnergyDistributionValue,
            "simulation parameters.cell.function.attacker.energy distribution value",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerColorInhomogeneityFactor,
            defaultParameters.cellFunctionAttackerColorInhomogeneityFactor,
            "simulation parameters.cell.function.attacker.color inhomogeneity factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerActivityThreshold,
            defaultParameters.cellFunctionAttackerActivityThreshold,
            "simulation parameters.cell.function.attacker.activity threshold",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerEnergyCost,
            defaultParameters.baseValues.cellFunctionAttackerEnergyCost,
            "simulation parameters.cell.function.attacker.energy cost",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
            defaultParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
            "simulation parameters.cell.function.attacker.geometry deviation exponent",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix,
            defaultParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix,
            "simulation parameters.cell.function.attacker.food chain color matrix",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
            defaultParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
            "simulation parameters.cell.function.attacker.connections mismatch penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus,
            defaultParameters.baseValues.cellFunctionAttackerGenomeComplexityBonus,
            "simulation parameters.cell.function.attacker.genome size bonus",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerSameMutantPenalty,
            defaultParameters.cellFunctionAttackerSameMutantPenalty,
            "simulation parameters.cell.function.attacker.same mutant penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty,
            defaultParameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty,
            "simulation parameters.cell.function.attacker.new complex mutant penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerSensorDetectionFactor,
            defaultParameters.cellFunctionAttackerSensorDetectionFactor,
            "simulation parameters.cell.function.attacker.sensor detection factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerDestroyCells,
            defaultParameters.cellFunctionAttackerDestroyCells,
            "simulation parameters.cell.function.attacker.destroy cells",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDefenderAgainstAttackerStrength,
            defaultParameters.cellFunctionDefenderAgainstAttackerStrength,
            "simulation parameters.cell.function.defender.against attacker strength",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDefenderAgainstInjectorStrength,
            defaultParameters.cellFunctionDefenderAgainstInjectorStrength,
            "simulation parameters.cell.function.defender.against injector strength",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionSameCreature,
            defaultParameters.cellFunctionTransmitterEnergyDistributionSameCreature,
            "simulation parameters.cell.function.transmitter.energy distribution same creature",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionRadius,
            defaultParameters.cellFunctionTransmitterEnergyDistributionRadius,
            "simulation parameters.cell.function.transmitter.energy distribution radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionValue,
            defaultParameters.cellFunctionTransmitterEnergyDistributionValue,
            "simulation parameters.cell.function.transmitter.energy distribution value",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleContractionExpansionDelta,
            defaultParameters.cellFunctionMuscleContractionExpansionDelta,
            "simulation parameters.cell.function.muscle.contraction expansion delta",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleMovementAcceleration,
            defaultParameters.cellFunctionMuscleMovementAcceleration,
            "simulation parameters.cell.function.muscle.movement acceleration",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleMovementAngleFromSensor,
            defaultParameters.cellFunctionMuscleMovementAngleFromSensor,
            "simulation parameters.cell.function.muscle.movement angle from sensor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleBendingAngle,
            defaultParameters.cellFunctionMuscleBendingAngle,
            "simulation parameters.cell.function.muscle.bending angle",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleBendingAcceleration,
            defaultParameters.cellFunctionMuscleBendingAcceleration,
            "simulation parameters.cell.function.muscle.bending acceleration",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleBendingAccelerationThreshold,
            defaultParameters.cellFunctionMuscleBendingAccelerationThreshold,
            "simulation parameters.cell.function.muscle.bending acceleration threshold",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.particleTransformationAllowed,
            defaultParameters.particleTransformationAllowed,
            "simulation parameters.particle.transformation allowed",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.particleTransformationRandomCellFunction,
            defaultParameters.particleTransformationRandomCellFunction,
            "simulation parameters.particle.transformation.random cell function",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.particleTransformationMaxGenomeSize,
            defaultParameters.particleTransformationMaxGenomeSize,
            "simulation parameters.particle.transformation.max genome size",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.particleSplitEnergy, defaultParameters.particleSplitEnergy, "simulation parameters.particle.split energy", parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionSensorRange,
            defaultParameters.cellFunctionSensorRange,
            "simulation parameters.cell.function.sensor.range",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionSensorActivityThreshold,
            defaultParameters.cellFunctionSensorActivityThreshold,
            "simulation parameters.cell.function.sensor.activity threshold",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionReconnectorRadius,
            defaultParameters.cellFunctionReconnectorRadius,
            "simulation parameters.cell.function.reconnector.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionReconnectorActivityThreshold,
            defaultParameters.cellFunctionReconnectorActivityThreshold,
            "simulation parameters.cell.function.reconnector.activity threshold",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDetonatorRadius,
            defaultParameters.cellFunctionDetonatorRadius,
            "simulation parameters.cell.function.detonator.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDetonatorChainExplosionProbability,
            defaultParameters.cellFunctionDetonatorChainExplosionProbability,
            "simulation parameters.cell.function.detonator.chain explosion probability",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDetonatorActivityThreshold,
            defaultParameters.cellFunctionDetonatorActivityThreshold,
            "simulation parameters.cell.function.detonator.activity threshold",
            parserTask);

        //particle sources
        encodeDecodeProperty(
            tree, parameters.numRadiationSources, defaultParameters.numRadiationSources, "simulation parameters.particle sources.num sources", parserTask);
        for (int index = 0; index < parameters.numRadiationSources; ++index) {
            std::string base = "simulation parameters.particle sources." + std::to_string(index) + ".";
            auto& source = parameters.radiationSources[index];
            auto& defaultSource = defaultParameters.radiationSources[index];
            encodeDecodeProperty(tree, source.posX, defaultSource.posX, base + "pos.x", parserTask);
            encodeDecodeProperty(tree, source.posY, defaultSource.posY, base + "pos.y", parserTask);
            encodeDecodeProperty(tree, source.velX, defaultSource.velX, base + "vel.x", parserTask);
            encodeDecodeProperty(tree, source.velY, defaultSource.velY, base + "vel.y", parserTask);
            encodeDecodeProperty(tree, source.useAngle, defaultSource.useAngle, base + "use angle", parserTask);
            encodeDecodeProperty(tree, source.angle, defaultSource.angle, base + "angle", parserTask);
            encodeDecodeProperty(tree, source.shapeType, defaultSource.shapeType, base + "shape.type", parserTask);
            if (source.shapeType == SpotShapeType_Circular) {
                encodeDecodeProperty(
                    tree,
                    source.shapeData.circularRadiationSource.radius,
                    defaultSource.shapeData.circularRadiationSource.radius,
                    base + "shape.circular.radius",
                    parserTask);
            }
            if (source.shapeType == SpotShapeType_Rectangular) {
                encodeDecodeProperty(
                    tree,
                    source.shapeData.rectangularRadiationSource.width,
                    defaultSource.shapeData.rectangularRadiationSource.width,
                    base + "shape.rectangular.width",
                    parserTask);
                encodeDecodeProperty(
                    tree,
                    source.shapeData.rectangularRadiationSource.height,
                    defaultSource.shapeData.rectangularRadiationSource.height,
                    base + "shape.rectangular.height",
                    parserTask);
            }
        }

        //spots
        encodeDecodeProperty(tree, parameters.numSpots, defaultParameters.numSpots, "simulation parameters.spots.num spots", parserTask);
        for (int index = 0; index < parameters.numSpots; ++index) {
            std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
            auto& spot = parameters.spots[index];
            auto& defaultSpot = defaultParameters.spots[index];
            encodeDecodeProperty(tree, spot.color, defaultSpot.color, base + "color", parserTask);
            encodeDecodeProperty(tree, spot.posX, defaultSpot.posX, base + "pos.x", parserTask);
            encodeDecodeProperty(tree, spot.posY, defaultSpot.posY, base + "pos.y", parserTask);
            encodeDecodeProperty(tree, spot.velX, defaultSpot.velX, base + "vel.x", parserTask);
            encodeDecodeProperty(tree, spot.velY, defaultSpot.velY, base + "vel.y", parserTask);

            encodeDecodeProperty(tree, spot.shapeType, defaultSpot.shapeType, base + "shape.type", parserTask);
            if (spot.shapeType == SpotShapeType_Circular) {
                encodeDecodeProperty(
                    tree,
                    spot.shapeData.circularSpot.coreRadius,
                    defaultSpot.shapeData.circularSpot.coreRadius,
                    base + "shape.circular.core radius",
                    parserTask);
            }
            if (spot.shapeType == SpotShapeType_Rectangular) {
                encodeDecodeProperty(
                    tree, spot.shapeData.rectangularSpot.width, defaultSpot.shapeData.rectangularSpot.width, base + "shape.rectangular.core width", parserTask);
                encodeDecodeProperty(
                    tree,
                    spot.shapeData.rectangularSpot.height,
                    defaultSpot.shapeData.rectangularSpot.height,
                    base + "shape.rectangular.core height",
                    parserTask);
            }
            encodeDecodeProperty(tree, spot.flowType, defaultSpot.flowType, base + "flow.type", parserTask);
            if (spot.flowType == FlowType_Radial) {
                encodeDecodeProperty(
                    tree, spot.flowData.radialFlow.orientation, defaultSpot.flowData.radialFlow.orientation, base + "flow.radial.orientation", parserTask);
                encodeDecodeProperty(
                    tree, spot.flowData.radialFlow.strength, defaultSpot.flowData.radialFlow.strength, base + "flow.radial.strength", parserTask);
                encodeDecodeProperty(
                    tree, spot.flowData.radialFlow.driftAngle, defaultSpot.flowData.radialFlow.driftAngle, base + "flow.radial.drift angle", parserTask);
            }
            if (spot.flowType == FlowType_Central) {
                encodeDecodeProperty(
                    tree, spot.flowData.centralFlow.strength, defaultSpot.flowData.centralFlow.strength, base + "flow.central.strength", parserTask);
            }
            if (spot.flowType == FlowType_Linear) {
                encodeDecodeProperty(tree, spot.flowData.linearFlow.angle, defaultSpot.flowData.linearFlow.angle, base + "flow.linear.angle", parserTask);
                encodeDecodeProperty(
                    tree, spot.flowData.linearFlow.strength, defaultSpot.flowData.linearFlow.strength, base + "flow.linear.strength", parserTask);
            }
            encodeDecodeProperty(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);

            encodeDecodeSpotProperty(tree, spot.values.friction, spot.activatedValues.friction, defaultSpot.values.friction, base + "friction", parserTask);
            encodeDecodeSpotProperty(tree, spot.values.rigidity, spot.activatedValues.rigidity, defaultSpot.values.rigidity, base + "rigidity", parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationDisableSources,
                spot.activatedValues.radiationDisableSources,
                defaultSpot.values.radiationDisableSources,
                base + "radiation.disable sources",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationAbsorption,
                spot.activatedValues.radiationAbsorption,
                defaultSpot.values.radiationAbsorption,
                base + "radiation.absorption",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationAbsorptionLowVelocityPenalty,
                spot.activatedValues.radiationAbsorptionLowVelocityPenalty,
                defaultSpot.values.radiationAbsorptionLowVelocityPenalty,
                base + "radiation.absorption low velocity penalty",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                spot.activatedValues.radiationAbsorptionLowGenomeComplexityPenalty,
                defaultSpot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                base +"radiation.absorption low genome complexity penalty",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationCellAgeStrength,
                spot.activatedValues.radiationCellAgeStrength,
                defaultSpot.values.radiationCellAgeStrength,
                base + "radiation.factor",
                parserTask);
            encodeDecodeSpotProperty(
                tree, spot.values.cellMaxForce, spot.activatedValues.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", parserTask);
            encodeDecodeSpotProperty(
                tree, spot.values.cellMinEnergy, spot.activatedValues.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", parserTask);

            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFusionVelocity,
                spot.activatedValues.cellFusionVelocity,
                defaultSpot.values.cellFusionVelocity,
                base + "cell.fusion velocity",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellMaxBindingEnergy,
                spot.activatedValues.cellMaxBindingEnergy,
                defaultSpot.values.cellMaxBindingEnergy,
                base + "cell.max binding energy",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellInactiveMaxAge,
                spot.activatedValues.cellInactiveMaxAge,
                defaultSpot.values.cellInactiveMaxAge,
                base + "cell.inactive max age",
                parserTask);

            encodeDecodeProperty(tree, spot.activatedValues.cellColorTransition, false, base + "cell.color transition rules.activated", parserTask);
            encodeDecodeProperty(
                tree,
                spot.values.cellColorTransitionDuration,
                defaultSpot.values.cellColorTransitionDuration,
                base + "cell.color transition rules.duration",
                parserTask);
            encodeDecodeProperty(
                tree,
                spot.values.cellColorTransitionTargetColor,
                defaultSpot.values.cellColorTransitionTargetColor,
                base + "cell.color transition rules.target color",
                parserTask);

            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerEnergyCost,
                spot.activatedValues.cellFunctionAttackerEnergyCost,
                defaultSpot.values.cellFunctionAttackerEnergyCost,
                base + "cell.function.attacker.energy cost",
                parserTask);

            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerFoodChainColorMatrix,
                spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix,
                defaultSpot.values.cellFunctionAttackerFoodChainColorMatrix,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerGenomeComplexityBonus,
                spot.activatedValues.cellFunctionAttackerGenomeComplexityBonus,
                defaultSpot.values.cellFunctionAttackerGenomeComplexityBonus,
                base + "cell.function.attacker.genome size bonus",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerNewComplexMutantPenalty,
                spot.activatedValues.cellFunctionAttackerNewComplexMutantPenalty,
                defaultSpot.values.cellFunctionAttackerNewComplexMutantPenalty,
                base + "cell.function.attacker.new complex mutant penalty",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerGeometryDeviationExponent,
                spot.activatedValues.cellFunctionAttackerGeometryDeviationExponent,
                defaultSpot.values.cellFunctionAttackerGeometryDeviationExponent,
                base + "cell.function.attacker.geometry deviation exponent",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerConnectionsMismatchPenalty,
                spot.activatedValues.cellFunctionAttackerConnectionsMismatchPenalty,
                defaultSpot.values.cellFunctionAttackerConnectionsMismatchPenalty,
                base + "cell.function.attacker.connections mismatch penalty",
                parserTask);

            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationNeuronData,
                spot.activatedValues.cellCopyMutationNeuronData,
                defaultSpot.values.cellCopyMutationNeuronData,
                base + "cell.copy mutation.neuron data",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationCellProperties,
                spot.activatedValues.cellCopyMutationCellProperties,
                defaultSpot.values.cellCopyMutationCellProperties,
                base + "cell.copy mutation.cell properties",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationGeometry,
                spot.activatedValues.cellCopyMutationGeometry,
                defaultSpot.values.cellCopyMutationGeometry,
                base + "cell.copy mutation.geometry",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationCustomGeometry,
                spot.activatedValues.cellCopyMutationCustomGeometry,
                defaultSpot.values.cellCopyMutationCustomGeometry,
                base + "cell.copy mutation.custom geometry",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationCellFunction,
                spot.activatedValues.cellCopyMutationCellFunction,
                defaultSpot.values.cellCopyMutationCellFunction,
                base + "cell.copy mutation.cell function",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationInsertion,
                spot.activatedValues.cellCopyMutationInsertion,
                defaultSpot.values.cellCopyMutationInsertion,
                base + "cell.copy mutation.insertion",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationDeletion,
                spot.activatedValues.cellCopyMutationDeletion,
                defaultSpot.values.cellCopyMutationDeletion,
                base + "cell.copy mutation.deletion",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationTranslation,
                spot.activatedValues.cellCopyMutationTranslation,
                defaultSpot.values.cellCopyMutationTranslation,
                base + "cell.copy mutation.translation",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationDuplication,
                spot.activatedValues.cellCopyMutationDuplication,
                defaultSpot.values.cellCopyMutationDuplication,
                base + "cell.copy mutation.duplication",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationCellColor,
                spot.activatedValues.cellCopyMutationCellColor,
                defaultSpot.values.cellCopyMutationCellColor,
                base + "cell.copy mutation.cell color",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationSubgenomeColor,
                spot.activatedValues.cellCopyMutationSubgenomeColor,
                defaultSpot.values.cellCopyMutationSubgenomeColor,
                base + "cell.copy mutation.subgenome color",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellCopyMutationGenomeColor,
                spot.activatedValues.cellCopyMutationGenomeColor,
                defaultSpot.values.cellCopyMutationGenomeColor,
                base + "cell.copy mutation.genome color",
                parserTask);
        }

        //features
        missingFeatures.genomeComplexityMeasurement = encodeDecodeProperty(
            tree,
            parameters.features.genomeComplexityMeasurement,
            defaultParameters.features.genomeComplexityMeasurement,
            "simulation parameters.features.genome complexity measurement",
            parserTask);
        missingFeatures.advancedAbsorptionControl = encodeDecodeProperty(
            tree,
            parameters.features.advancedAbsorptionControl,
            defaultParameters.features.advancedAbsorptionControl,
            "simulation parameters.features.additional absorption control",
            parserTask);
        missingFeatures.advancedAttackerControl = encodeDecodeProperty(
            tree,
            parameters.features.advancedAttackerControl,
            defaultParameters.features.advancedAttackerControl,
            "simulation parameters.features.additional attacker control",
            parserTask);
        missingFeatures.advancedMuscleControl = encodeDecodeProperty(
            tree,
            parameters.features.advancedMuscleControl,
            defaultParameters.features.advancedMuscleControl,
            "simulation parameters.features.additional muscle control",
            parserTask);
        missingFeatures.externalEnergyControl = encodeDecodeProperty(
            tree, parameters.features.externalEnergyControl, defaultParameters.features.externalEnergyControl, "simulation parameters.features.external energy", parserTask);
        missingFeatures.cellColorTransitionRules = encodeDecodeProperty(
            tree,
            parameters.features.cellColorTransitionRules,
            defaultParameters.features.cellColorTransitionRules,
            "simulation parameters.features.cell color transition rules",
            parserTask);
        missingFeatures.cellAgeLimiter = encodeDecodeProperty(
            tree,
            parameters.features.cellAgeLimiter,
            defaultParameters.features.cellAgeLimiter,
            "simulation parameters.features.cell age limiter",
            parserTask);
        missingFeatures.cellGlow = encodeDecodeProperty(
            tree,
            parameters.features.cellGlow,
            defaultParameters.features.cellGlow,
            "simulation parameters.features.cell glow",
            parserTask);
    }

    void encodeDecodeSimulationParameters(boost::property_tree::ptree& tree, SimulationParameters& parameters, ParserTask parserTask)
    {
        MissingParameters missingParameters;
        Features missingFeatures;
        encodeDecode(tree, parameters, missingParameters, missingFeatures, parserTask);

        // Compatibility with legacy parameters
        if (parserTask == ParserTask::Decode) {
            searchAndApplyLegacyParameters(tree, missingFeatures, missingParameters, parameters);
        }
    }

    void encodeDecode(boost::property_tree::ptree& tree, AuxiliaryData& data, ParserTask parserTask)
    {
        AuxiliaryData defaultSettings;

        //general settings
        encodeDecodeProperty(tree, data.timestep, uint64_t(0), "general.time step", parserTask);
        encodeDecodeProperty(tree, data.realTime, std::chrono::milliseconds(0), "general.real time", parserTask);
        encodeDecodeProperty(tree, data.zoom, 4.0f, "general.zoom", parserTask);
        encodeDecodeProperty(tree, data.center.x, 0.0f, "general.center.x", parserTask);
        encodeDecodeProperty(tree, data.center.y, 0.0f, "general.center.y", parserTask);
        encodeDecodeProperty(tree, data.generalSettings.worldSizeX, defaultSettings.generalSettings.worldSizeX, "general.world size.x", parserTask);
        encodeDecodeProperty(tree, data.generalSettings.worldSizeY, defaultSettings.generalSettings.worldSizeY, "general.world size.y", parserTask);

        encodeDecodeSimulationParameters(tree, data.simulationParameters, parserTask);
    }
}

boost::property_tree::ptree AuxiliaryDataParserService::encodeAuxiliaryData(AuxiliaryData const& data)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, const_cast<AuxiliaryData&>(data), ParserTask::Encode);
    return tree;
}

AuxiliaryData AuxiliaryDataParserService::decodeAuxiliaryData(boost::property_tree::ptree tree)
{
    AuxiliaryData result;
    encodeDecode(tree, result, ParserTask::Decode);
    return result;
}

boost::property_tree::ptree AuxiliaryDataParserService::encodeSimulationParameters(SimulationParameters const& data)
{
    boost::property_tree::ptree tree;
    encodeDecodeSimulationParameters(tree, const_cast<SimulationParameters&>(data), ParserTask::Encode);
    return tree;
}

SimulationParameters AuxiliaryDataParserService::decodeSimulationParameters(boost::property_tree::ptree tree)
{
    SimulationParameters result;
    encodeDecodeSimulationParameters(tree, result, ParserTask::Decode);
    return result;
}
