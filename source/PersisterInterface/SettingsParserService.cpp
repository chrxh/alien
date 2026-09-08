#include "SettingsParserService.h"

#include <sstream>

#include <boost/property_tree/json_parser.hpp>

#include <Base/Resources.h>

#include <EngineInterface/SimulationParametersSpecification.h>
#include <EngineInterface/SpecificationEvaluationService.h>

#include "ParameterParser.h"

namespace
{
    template <typename Spec>
    void encodeDecodeParameterForLocation(
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters,
        SimulationParameters& defaultParameters,
        int orderNumber,
        Spec& spec,
        ParserTask parserTask,
        std::string const& nodeBase)
    {
        auto& evaluationService = SpecificationEvaluationService::get();

        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        auto defaultOrderNumber = [&] {
            if (locationType == LocationType::Base) {
                return 0;
            } else if (locationType == LocationType::Layer) {
                return defaultParameters.layerOrderNumbers[0];  // Default layer
            } else if (locationType == LocationType::Source) {
                return defaultParameters.sourceOrderNumbers[0];  // Default source
            } else {
                CHECK(false);
            }
        }();

        auto ref = evaluationService.getRef(spec._member, parameters, orderNumber);
        auto defaultRef = evaluationService.getRef(spec._member, defaultParameters, defaultOrderNumber);

        if (ref.value) {
            if (ref.colorDependence == ColorDependence::None) {
                ParameterParser::encodeDecode(tree, *ref.value, *defaultRef.value, nodeBase + ".Value", parserTask);
            } else if (ref.colorDependence == ColorDependence::ColorVector) {
                for (int i = 0; i < MAX_COLORS; ++i) {
                    ParameterParser::encodeDecode(tree, ref.value[i], defaultRef.value[i], nodeBase + ".Color " + std::to_string(i), parserTask);
                }
            } else if (ref.colorDependence == ColorDependence::ColorMatrix) {
                for (int i = 0; i < MAX_COLORS; ++i) {
                    for (int j = 0; j < MAX_COLORS; ++j) {
                        ParameterParser::encodeDecode(
                            tree,
                            ref.value[i * MAX_COLORS + j],
                            defaultRef.value[i * MAX_COLORS + j],
                            nodeBase + ".Color " + std::to_string(i) + "," + std::to_string(j),
                            parserTask);
                    }
                }
            }
        }
        if (ref.enabled) {
            ParameterParser::encodeDecode(tree, *ref.enabled, *defaultRef.enabled, nodeBase + ".Enabled", parserTask);
        }
        if (ref.pinned) {
            ParameterParser::encodeDecode(tree, *ref.pinned, *defaultRef.pinned, nodeBase + ".Pinned", parserTask);
        }
    }

    template <typename Spec>
    void encodeDecodeParameter(
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters,
        SimulationParameters& defaultParameters,
        Spec& spec,
        ParserTask parserTask,
        std::string const& nodeBase)
    {
        encodeDecodeParameterForLocation(tree, parameters, defaultParameters, 0, spec, parserTask, nodeBase + ".Base");
        for (int i = 0; i < parameters.numLayers; ++i) {
            encodeDecodeParameterForLocation(
                tree, parameters, defaultParameters, parameters.layerOrderNumbers[i], spec, parserTask, nodeBase + ".Layer " + std::to_string(i));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            encodeDecodeParameterForLocation(
                tree, parameters, defaultParameters, parameters.sourceOrderNumbers[i], spec, parserTask, nodeBase + ".Source " + std::to_string(i));
        }
    }

    void encodeDecodeSimulationParameterGroup(
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters,
        SimulationParameters& defaultParameters,
        std::string const& nodeBase,
        ParserTask parserTask,
        std::vector<ParameterSpec> const& parameterSpecs)
    {
        for (auto const& parameterSpec : parameterSpecs) {
            if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<BoolSpec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
            } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<IntSpec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
            } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<FloatSpec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
            } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<Float2Spec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
            } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<Char64Spec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
            } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
                auto const& altSpec = std::get<AlternativeSpec>(parameterSpec._reference);
                encodeDecodeParameter(tree, parameters, defaultParameters, altSpec, parserTask, nodeBase + "." + parameterSpec._name);
                for (auto const& [alternative, parameterSpecs] : altSpec._alternatives) {
                    encodeDecodeSimulationParameterGroup(
                        tree, parameters, defaultParameters, nodeBase + "." + parameterSpec._name + "." + alternative, parserTask, parameterSpecs);
                }
            } else if (std::holds_alternative<ColorSpec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<ColorSpec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
            } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree,
                    parameters,
                    defaultParameters,
                    std::get<ColorTransitionRulesSpec>(parameterSpec._reference),
                    parserTask,
                    nodeBase + "." + parameterSpec._name);
            }
        }
    }

    void
    encodeDecodeSimulationParameters(boost::property_tree::ptree& tree, SimulationParameters& parameters, std::string const& nodeBase, ParserTask parserTask)
    {
        auto& evaluationService = SpecificationEvaluationService::get();

        SimulationParameters defaultParameters;
        defaultParameters.numLayers = 1;
        defaultParameters.numSources = 1;
        defaultParameters.layerOrderNumbers[0] = 1;
        defaultParameters.sourceOrderNumbers[0] = 2;

        ParameterParser::encodeDecode(tree, parameters.numLayers, 0, nodeBase + ".Number of layers", parserTask);
        ParameterParser::encodeDecode(tree, parameters.numSources, 0, nodeBase + ".Number of sources", parserTask);
        for (int i = 0; i < parameters.numLayers; ++i) {
            ParameterParser::encodeDecode(
                tree,
                parameters.layerOrderNumbers[i],
                defaultParameters.layerOrderNumbers[i],
                nodeBase + ".Layer order number.Index " + std::to_string(i),
                parserTask);
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            ParameterParser::encodeDecode(
                tree,
                parameters.sourceOrderNumbers[i],
                defaultParameters.sourceOrderNumbers[i],
                nodeBase + ".Source order number.Index " + std::to_string(i),
                parserTask);
        }

        auto const& parametersSpecs = SimulationParameters::getSpec();
        for (auto const& groupSpec : parametersSpecs._groups) {

            if (groupSpec._expertToggle != nullptr) {
                auto expertToggleRef = evaluationService.getExpertToggleRef(groupSpec._expertToggle, parameters);
                auto defaultExpertToggleRef = evaluationService.getExpertToggleRef(groupSpec._expertToggle, defaultParameters);
                ParameterParser::encodeDecode(tree, *expertToggleRef, *defaultExpertToggleRef, nodeBase + "." + groupSpec._name + ".Enabled", parserTask);
            }
            encodeDecodeSimulationParameterGroup(tree, parameters, defaultParameters, nodeBase + "." + groupSpec._name, parserTask, groupSpec._parameters);
        }
    }

    auto const GeneralNode = std::string("General");
    auto const SimulationParametersNode = std::string("Simulation parameters");
}

boost::property_tree::ptree SettingsParserService::encodeSimulationParameters(SimulationParameters const& data)
{
    boost::property_tree::ptree tree;
    auto programVersion = Const::ProgramVersion;
    ParameterParser::encodeDecode(tree, programVersion, std::string(), GeneralNode + ".Version", ParserTask::Encode);
    encodeDecodeSimulationParameters(tree, const_cast<SimulationParameters&>(data), SimulationParametersNode, ParserTask::Encode);
    return tree;
}

// BEGIN: Temporary compatibility code for reading the legacy simulation parameters format. Remove after all simulations have been migrated.
namespace
{
    // The legacy inflow parameter was a fraction of this reference energy instead of an absolute energy.
    auto constexpr LegacyInflowReferenceEnergy = 50.0f;

    void migrateLegacyExternalEnergyGroup(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        auto legacyGroup = tree.get_child_optional(nodeBase + ".External energy control");
        if (!legacyGroup || tree.get_child_optional(nodeBase + ".Guided energy supply")) {
            return;
        }
        auto& group = tree.put_child(nodeBase + ".Guided energy supply", *legacyGroup);

        boost::property_tree::ptree inflowForConstructor;
        for (int color = 0; color < MAX_COLORS; ++color) {
            auto colorNode = "Color " + std::to_string(color);
            if (auto inflowFactor = group.get_optional<float>("Inflow.Base." + colorNode)) {
                inflowForConstructor.put(colorNode, *inflowFactor * LegacyInflowReferenceEnergy);
            }
        }
        group.put_child("Inflow for constructors.Base", inflowForConstructor);

        // The legacy expert toggle switched off the whole external energy handling.
        if (auto enabled = group.get_optional<bool>("Enabled"); enabled && !*enabled) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                auto colorNode = "Color " + std::to_string(color);
                group.put("Inflow for constructors.Base." + colorNode, 0.0f);
                group.put("Inflow for sources.Base." + colorNode, 0.0f);
                group.put("Backflow.Base." + colorNode, 0.0f);
            }
        }
    }

    void migrateLegacySimulationParameters(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        migrateLegacyExternalEnergyGroup(tree, nodeBase);

        if (auto transformationAllowed = tree.get_optional<std::string>(nodeBase + ".Radiation.Energy to cell transformation.Base.Value")) {
            tree.put(nodeBase + ".Cell life cycle.Energy to cell free transformation.Base.Value", *transformationAllowed);
        }
    }
}
// END: Temporary compatibility code

SimulationParameters SettingsParserService::decodeSimulationParameters(boost::property_tree::ptree tree)
{
    migrateLegacySimulationParameters(tree, SimulationParametersNode);

    SimulationParameters result;
    encodeDecodeSimulationParameters(tree, result, SimulationParametersNode, ParserTask::Decode);
    return result;
}

std::string SettingsParserService::encodeSimulationParametersToString(SimulationParameters const& data)
{
    std::stringstream stream;
    boost::property_tree::json_parser::write_json(stream, encodeSimulationParameters(data));
    return stream.str();
}

SimulationParameters SettingsParserService::decodeSimulationParametersFromString(std::string const& data)
{
    std::stringstream stream(data);
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    return decodeSimulationParameters(tree);
}
