#include "SettingsParserService.h"

#include <ranges>

#include "Base/Resources.h"
#include "EngineInterface/SimulationParametersSpecification.h"
#include "EngineInterface/SpecificationEvaluationService.h"

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
            } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
                encodeDecodeParameter(
                    tree, parameters, defaultParameters, std::get<ColorPickerSpec>(parameterSpec._reference), parserTask, nodeBase + "." + parameterSpec._name);
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

    void encodeDecodeSimulationParameters(boost::property_tree::ptree& tree, SimulationParameters& parameters, std::string const& nodeBase, ParserTask parserTask)
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

    void encodeDecode(boost::property_tree::ptree& tree, SettingsForSerialization& data, ParserTask parserTask)
    {
        SettingsForSerialization defaultSettings;

        // General settings
        auto nodeBase = std::string("General");
        auto programVersion = Const::ProgramVersion;
        ParameterParser::encodeDecode(tree, programVersion, std::string(), nodeBase + ".Version", parserTask);
        ParameterParser::encodeDecode(tree, data.timestep, defaultSettings.timestep, nodeBase + ".Time step", parserTask);
        ParameterParser::encodeDecode(tree, data.realTime, defaultSettings.realTime, nodeBase + ".Real time", parserTask);
        ParameterParser::encodeDecode(tree, data.zoom, defaultSettings.zoom, nodeBase + ".Zoom", parserTask);
        ParameterParser::encodeDecode(tree, data.center, defaultSettings.center, nodeBase + ".Center", parserTask);
        ParameterParser::encodeDecode(tree, data.worldSize, defaultSettings.worldSize, nodeBase + ".World size", parserTask);

        encodeDecodeSimulationParameters(tree, data.simulationParameters, "Simulation parameters", parserTask);
    }
}

boost::property_tree::ptree SettingsParserService::encodeSettings(SettingsForSerialization const& data)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, const_cast<SettingsForSerialization&>(data), ParserTask::Encode);
    return tree;
}

SettingsForSerialization SettingsParserService::decodeSettings(boost::property_tree::ptree tree)
{
    SettingsForSerialization result;
    encodeDecode(tree, result, ParserTask::Decode);
    return result;
}

boost::property_tree::ptree SettingsParserService::encodeSimulationParameters(SimulationParameters const& data)
{
    boost::property_tree::ptree tree;
    encodeDecodeSimulationParameters(tree, const_cast<SimulationParameters&>(data), "Simulation parameters", ParserTask::Encode);
    return tree;
}

SimulationParameters SettingsParserService::decodeSimulationParameters(boost::property_tree::ptree tree)
{
    SimulationParameters result;
    encodeDecodeSimulationParameters(tree, result, "Simulation parameters", ParserTask::Decode);
    return result;
}
