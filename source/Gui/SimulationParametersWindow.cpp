#include "SimulationParametersWindow.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "RadiationSourcesWindow.h"
#include "SimulationParametersChanger.h"

namespace
{
    auto const RightColumnWidth = 260.0f;

    template <int numRows, int numCols>
    std::vector<std::vector<float>> toVector(float const v[numRows][numCols])
    {
        std::vector<std::vector<float>> result;
        for (int row = 0; row < numRows; ++row) {
            std::vector<float> rowVector;
            for (int col = 0; col < numCols; ++col) {
                rowVector.emplace_back(v[row][col]);
            }
            result.emplace_back(rowVector);
        }
        return result;
    }

    template<int numElements>
    std::vector<float> toVector(float const v[numElements])
    {
        std::vector<float> result;
        for (int i = 0; i < numElements; ++i) {
            result.emplace_back(v[i]);
        }
        return result;
    }
}

_SimulationParametersWindow::_SimulationParametersWindow(SimulationController const& simController, RadiationSourcesWindow const& radiationSourcesWindow)
    : _AlienWindow("Simulation parameters", "windows.simulation parameters", false)
    , _simController(simController)
    , _radiationSourcesWindow(radiationSourcesWindow)
{
    for (int n = 0; n < IM_ARRAYSIZE(_savedPalette); n++) {
        ImVec4 color;
        ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.2f, color.x, color.y, color.z);
        color.w = 1.0f; //alpha
        _savedPalette[n] = static_cast<ImU32>(ImColor(color));
    }

    auto timestepsPerEpoch = GlobalSettings::getInstance().getIntState("windows.simulation parameters.time steps per epoch", 10000);

    _simulationParametersChanger = std::make_shared<_SimulationParametersChanger>(simController, timestepsPerEpoch);
}

_SimulationParametersWindow::~_SimulationParametersWindow()
{
    GlobalSettings::getInstance().setIntState("windows.simulation parameters.time steps per epoch", _simulationParametersChanger->getTimestepsPerEpoch());
}

void _SimulationParametersWindow::processIntern()
{
    auto parameters = _simController->getSimulationParameters();
    auto origParameters = _simController->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(78)), false)) {

        if (ImGui::BeginTabBar("##Flow", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

            //add spot
            if (parameters.numSpots < MAX_SPOTS) {
                if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                    int index = parameters.numSpots;
                    parameters.spots[index] = createSpot(parameters, index);
                    origParameters.spots[index] = createSpot(parameters, index);
                    ++parameters.numSpots;
                    ++origParameters.numSpots;
                    _simController->setSimulationParameters_async(parameters);
                    _simController->setOriginalSimulationParameters(origParameters);
                }
                AlienImGui::Tooltip("Add spot");
            }

            if (ImGui::BeginTabItem("Base", nullptr, ImGuiTabItemFlags_None)) {
                processBase(parameters, origParameters);
                ImGui::EndTabItem();
            }

            for (int tab = 0; tab < parameters.numSpots; ++tab) {
                SimulationParametersSpot& spot = parameters.spots[tab];
                SimulationParametersSpot const& origSpot = origParameters.spots[tab];
                bool open = true;
                std::string name = "Spot " + std::to_string(tab+1);
                if (ImGui::BeginTabItem(name.c_str(), &open, ImGuiTabItemFlags_None)) {
                    processSpot(spot, origSpot);
                    ImGui::EndTabItem();
                }

                //delete spot
                if (!open) {
                    for (int i = tab; i < parameters.numSpots - 1; ++i) {
                        parameters.spots[i] = parameters.spots[i + 1];
                        origParameters.spots[i] = origParameters.spots[i + 1];
                    }
                    --parameters.numSpots;
                    --origParameters.numSpots;
                    _simController->setSimulationParameters_async(parameters);
                    _simController->setOriginalSimulationParameters(origParameters);
                }
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Change automatically"), _changeAutomatically)) {
        if (_changeAutomatically) {
            _simulationParametersChanger->activate();
        } else {
            _simulationParametersChanger->deactivate();
        }
    }
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::BeginDisabled(!_changeAutomatically);
    auto timestepsPerEpoch = _simulationParametersChanger->getTimestepsPerEpoch();
    if (AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Epoch time steps")
                .defaultValue(_simulationParametersChanger->getOriginalTimestepsPerEpoch())
                .textWidth(RightColumnWidth)
                .tooltip("Duration in time steps after which a change is applied."),
            timestepsPerEpoch)) {
        _simulationParametersChanger->setTimestepsPerEpoch(timestepsPerEpoch);
    }
    ImGui::EndDisabled();

    if (parameters != lastParameters) {
        _simController->setSimulationParameters_async(parameters);
    }
}

void _SimulationParametersWindow::processBackground()
{
    _simulationParametersChanger->process();
}

SimulationParametersSpot _SimulationParametersWindow::createSpot(SimulationParameters const& simParameters, int index)
{
    auto worldSize = _simController->getWorldSize();
    SimulationParametersSpot spot;
    spot.posX = toFloat(worldSize.x / 2);
    spot.posY = toFloat(worldSize.y / 2);

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    spot.shapeType = ShapeType_Circular;
    createDefaultSpotData(spot);
    spot.fadeoutRadius = maxRadius / 3;
    spot.color = _savedPalette[((2 + index) * 8) % IM_ARRAYSIZE(_savedPalette)];

    spot.values = simParameters.baseValues;
    return spot;
}

void _SimulationParametersWindow::createDefaultSpotData(SimulationParametersSpot& spot)
{
    auto worldSize = _simController->getWorldSize();

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    if (spot.shapeType == ShapeType_Circular) {
        spot.shapeData.circularSpot.coreRadius = maxRadius / 3;
    } else {
        spot.shapeData.rectangularSpot.height = maxRadius / 3;
        spot.shapeData.rectangularSpot.width = maxRadius / 3;
    }
}

void _SimulationParametersWindow::processBase(
    SimulationParameters& simParameters,
    SimulationParameters const& origSimParameters)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;

        /**
         * Colors
         */
        if (ImGui::TreeNodeEx("Colors", flags)) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSimParameters.backgroundColor),
                simParameters.backgroundColor,
                _backupColor,
                _savedPalette);
            ImGui::TreePop();
        }

        /**
         * Numerics
         */
        if (ImGui::TreeNodeEx("Numerics", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Time step size")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.timestepSize)
                    .tooltip(std::string("Time duration calculated in a single step. Smaller values increase the accuracy "
                                         "of the simulation.")),
                simParameters.timestepSize);
            ImGui::TreePop();
        }

        /**
         * Physics: General
         */
        if (ImGui::TreeNodeEx("Physics: General", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Friction")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.4f")
                    .defaultValue(origSimParameters.baseValues.friction)
                    .tooltip(std::string("Specifies how much the movements are slowed down per time step.")),
                simParameters.baseValues.friction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Rigidity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .format("%.2f")
                    .defaultValue(origSimParameters.baseValues.rigidity)
                    .tooltip(std::string("Controls the rigidity of connected cells.\nA higher value will cause connected cells to move more uniformly.")),
                simParameters.baseValues.rigidity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum velocity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(6.0f)
                    .defaultValue(origSimParameters.cellMaxVelocity)
                    .tooltip(std::string("Maximum velocity that a cell can reach.")),
                simParameters.cellMaxVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum force")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .defaultValue(origSimParameters.baseValues.cellMaxForce)
                    .tooltip(std::string("Maximum force that can be applied to a cell without causing it to disintegrate.")),
                simParameters.baseValues.cellMaxForce);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum distance")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellMinDistance)
                    .tooltip(std::string("Minimum distance between two cells without them annihilating each other.")),
                simParameters.cellMinDistance);
            ImGui::TreePop();
        }

        /**
         * Physics: Radiation
         */
        if (ImGui::TreeNodeEx("Physics: Radiation", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radiation strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .format("%.6f")
                    .defaultValue(origSimParameters.baseValues.radiationFactor)
                    .tooltip(std::string("Indicates how energetic the emitted particles of cells are.")),
                simParameters.baseValues.radiationFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum energy")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(100000)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.radiationMinCellEnergy)
                    .tooltip(""),
                simParameters.radiationMinCellEnergy);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Minimum age")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1000000)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.radiationMinCellAge)
                    .tooltip(""),
                simParameters.radiationMinCellAge);
            AlienImGui::InputColorVector(
                AlienImGui::InputColorVectorParameters()
                    .name("Absorption factors")
                    .textWidth(RightColumnWidth)
                    .defaultValue(toVector<MAX_COLORS>(origSimParameters.radiationAbsorptionByCellColor))
                    .tooltip(""),
                simParameters.radiationAbsorptionByCellColor);
            if(AlienImGui::Button(
                AlienImGui::ButtonParameters().buttonText("Define").name("Particle sources editor").textWidth(RightColumnWidth).tooltip(""))) {
                _radiationSourcesWindow->setOn(true);
            }
            ImGui::TreePop();
        }

        /**
         * Physics: Particle transformation
         */
        ImGui::PushID("Transformation");
        if (ImGui::TreeNodeEx("Physics: Particle transformation", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum energy")
                    .textWidth(RightColumnWidth)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSimParameters.baseValues.cellMinEnergy)
                    .tooltip(std::string("Minimum energy a cell needs to exist.")),
                simParameters.baseValues.cellMinEnergy);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Normal energy")
                    .textWidth(RightColumnWidth)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSimParameters.cellNormalEnergy),
                simParameters.cellNormalEnergy);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters().name("Cell cluster decay").textWidth(RightColumnWidth).defaultValue(origSimParameters.clusterDecay),
                simParameters.clusterDecay);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell cluster decay probability")
                    .textWidth(RightColumnWidth)
                    .min(0.0)
                    .max(1.0f)
                    .format("%.5f")
                    .defaultValue(origSimParameters.clusterDecayProb),
                simParameters.clusterDecayProb);
            ImGui::TreePop();
        }
        ImGui::PopID();

        /**
         * Physics: Collision and binding
         */
        if (ImGui::TreeNodeEx("Physics: Collision and binding", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Repulsion strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.3f)
                    .defaultValue(origSimParameters.cellRepulsionStrength)
                    .tooltip(std::string("The strength of the repulsive forces, between two cells that do not connect.")),
                simParameters.cellRepulsionStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum collision distance")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellMaxCollisionDistance)
                    .tooltip(std::string("Maximum distance up to which a collision of two cells is possible.")),
                simParameters.cellMaxCollisionDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum binding distance")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellMaxBindingDistance)
                    .tooltip(std::string("Maximum distance up to which a connection of two cells is possible.")),
                simParameters.cellMaxBindingDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Binding creation velocity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.baseValues.cellFusionVelocity)
                    .tooltip(std::string("Minimum velocity of two colliding cells so that a connection can be established.")),
                simParameters.baseValues.cellFusionVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Binding maximum energy")
                    .textWidth(RightColumnWidth)
                    .min(50.0f)
                    .max(1000000.0f)
                    .logarithmic(true)
                    .format("%.0f")
                    .defaultValue(origSimParameters.baseValues.cellMaxBindingEnergy)
                    .tooltip(std::string("Maximum energy of a cell at which they can maintain a connection.")),
                simParameters.baseValues.cellMaxBindingEnergy);
            if (simParameters.baseValues.cellMaxBindingEnergy < simParameters.baseValues.cellMinEnergy + 10.0f) {
                simParameters.baseValues.cellMaxBindingEnergy = simParameters.baseValues.cellMinEnergy + 10.0f;
            }
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Maximum cell bonds")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellMaxBonds)
                    .min(0)
                    .max(6)
                    .tooltip(std::string("Maximum number of connections a cell can establish with others.")),
                simParameters.cellMaxBonds);
            ImGui::TreePop();
        }

        /**
         * Cell color transition rules
         */
        if (ImGui::TreeNodeEx("Cell color transition rules", flags)) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                ImGui::PushID(color);
                auto parameters = AlienImGui::InputColorTransitionParameters()
                                      .textWidth(RightColumnWidth)
                                      .color(color)
                                      .defaultTargetColor(origSimParameters.baseValues.cellColorTransitionTargetColor[color])
                                      .defaultTransitionAge(origSimParameters.baseValues.cellColorTransitionDuration[color])
                                      .logarithmic(true);
                if (0 == color) {
                    parameters.name("Target color and duration")
                        .tooltip("Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent color can "
                                 "be defined for each cell color. In addition, durations must be specified that define how many time steps the corresponding "
                                 "color are kept.");
                }
                AlienImGui::InputColorTransition(
                    parameters,
                    color,
                    simParameters.baseValues.cellColorTransitionTargetColor[color],
                    simParameters.baseValues.cellColorTransitionDuration[color]);
                ImGui::PopID();
            }
            ImGui::TreePop();
        }

        /**
         * Mutation 
         */
        if (ImGui::TreeNodeEx("Cell function: Genome mutation probabilities", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neuron weights and bias")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability),
                simParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDataProbability),
                simParameters.baseValues.cellFunctionConstructorMutationDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability),
                simParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationInsertionProbability),
                simParameters.baseValues.cellFunctionConstructorMutationInsertionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDeletionProbability),
                simParameters.baseValues.cellFunctionConstructorMutationDeletionProbability);
            ImGui::TreePop();
        }

        /**
         * Constructor
         */
        if (ImGui::TreeNodeEx("Cell function: Constructor", flags)) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Offspring inherit color")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionConstructionInheritColor),
                simParameters.cellFunctionConstructionInheritColor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Offspring distance")
                    .textWidth(RightColumnWidth)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorOffspringDistance),
                simParameters.cellFunctionConstructorOffspringDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Offspring connection distance")
                    .textWidth(RightColumnWidth)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorConnectingCellMaxDistance),
                simParameters.cellFunctionConstructorConnectingCellMaxDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Activity threshold")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorActivityThreshold),
                simParameters.cellFunctionConstructorActivityThreshold);
            ImGui::TreePop();
        }

        /**
         * Attacker
         */
        ImGui::PushID("Attacker");
        if (ImGui::TreeNodeEx("Cell function: Attacker", flags)) {
            AlienImGui::InputColorMatrix(
                AlienImGui::InputColorMatrixParameters()
                    .name("Food chain color matrix")
                    .textWidth(RightColumnWidth)
                    .tooltip(
                        "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell is shown in the "
                        "header "
                        "column while the color of the attacked cell is shown in the header row. A value of 0 means that the attacked cell cannot be digested, "
                        "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: If a "
                        "0 "
                        "is "
                        "entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells.")
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix)),
                simParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy cost")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerEnergyCost)
                    .tooltip(std::string("Amount of energy lost by an attempted attack of a cell in the form of emitted energy particles.")),
                simParameters.baseValues.cellFunctionAttackerEnergyCost);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry penalty")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent)
                    .tooltip(std::string("The larger this value is, the less energy a cell can gain from an attack if the local "
                                         "geometry of the attacked cell does not match the attacking cell.")),
                simParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connections mismatch penalty")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty)
                    .tooltip(std::string("The larger this parameter is, the more difficult it is to digest cells that contain more connections.")),
                simParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Attack radius")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(2.5f)
                    .defaultValue(origSimParameters.cellFunctionAttackerRadius),
                simParameters.cellFunctionAttackerRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Attack strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.1f)
                    .defaultValue(origSimParameters.cellFunctionAttackerStrength),
                simParameters.cellFunctionAttackerStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution radius")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionRadius),
                simParameters.cellFunctionAttackerEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionValue),
                simParameters.cellFunctionAttackerEnergyDistributionValue);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Same color energy distribution")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionSameColor),
                simParameters.cellFunctionAttackerEnergyDistributionSameColor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color inhomogeneity factor")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerColorInhomogeneityFactor),
                simParameters.cellFunctionAttackerColorInhomogeneityFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Activity threshold")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerActivityThreshold),
                simParameters.cellFunctionAttackerActivityThreshold);
            ImGui::TreePop();
        }
        ImGui::PopID();

        /**
         * Transmitter
         */
        if (ImGui::TreeNodeEx("Cell function: Transmitter", flags)) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Same color energy distribution")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionSameColor),
                simParameters.cellFunctionTransmitterEnergyDistributionSameColor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution radius")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionRadius),
                simParameters.cellFunctionTransmitterEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionValue),
                simParameters.cellFunctionTransmitterEnergyDistributionValue);
            ImGui::TreePop();
        }

        /**
         * Muscle
         */
        if (ImGui::TreeNodeEx("Cell function: Muscle", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Contraction and expansion delta")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.1f)
                    .defaultValue(origSimParameters.cellFunctionMuscleContractionExpansionDelta),
                simParameters.cellFunctionMuscleContractionExpansionDelta);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Acceleration")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.15f)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.cellFunctionMuscleMovementAcceleration),
                simParameters.cellFunctionMuscleMovementAcceleration);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Bending angle")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(10.0f)
                    .defaultValue(origSimParameters.cellFunctionMuscleBendingAngle),
                simParameters.cellFunctionMuscleBendingAngle);
            ImGui::TreePop();
        }

        /**
         * Sensor
         */
        if (ImGui::TreeNodeEx("Cell function: Sensor", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Range")
                    .textWidth(RightColumnWidth)
                    .min(10.0f)
                    .max(512.0f)
                    .defaultValue(origSimParameters.cellFunctionSensorRange)
                    .tooltip(std::string("The maximum radius in which a sensor can detect mass concentrations.")),
                simParameters.cellFunctionSensorRange);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Activity threshold")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellFunctionSensorActivityThreshold),
                simParameters.cellFunctionSensorActivityThreshold);
            ImGui::TreePop();
        }

        /**
         * Danger zone
         */
        if (ImGui::TreeNodeEx("Danger zone", flags)) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Unlimited energy for constructor")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionConstructionUnlimitedEnergy),
                simParameters.cellFunctionConstructionUnlimitedEnergy);
            ImGui::TreePop();
        }
    }
    ImGui::EndChild();
    validationAndCorrection(simParameters);
}

void _SimulationParametersWindow::processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;
        auto worldSize = _simController->getWorldSize();

        /**
         * Colors and location
         */
        if (ImGui::TreeNodeEx("Colors and location", flags)) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSpot.color),
                spot.color,
                _backupColor,
                _savedPalette);

            if (AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Shape")
                    .values({"Circular", "Rectangular"})
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSpot.shapeType),
                spot.shapeType)) {
                createDefaultSpotData(spot);
            }
            auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y));
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Position X")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(toFloat(worldSize.x))
                    .defaultValue(origSpot.posX)
                    .format("%.1f"),
                spot.posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Position Y")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(toFloat(worldSize.y))
                    .defaultValue(origSpot.posY)
                    .format("%.1f"),
                spot.posY);
            if (spot.shapeType == ShapeType_Circular) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Core radius")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(maxRadius)
                        .defaultValue(origSpot.shapeData.circularSpot.coreRadius)
                        .format("%.1f"),
                    spot.shapeData.circularSpot.coreRadius);
            }
            if (spot.shapeType == ShapeType_Rectangular) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Core width")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(worldSize.x)
                        .defaultValue(origSpot.shapeData.rectangularSpot.width)
                        .format("%.1f"),
                    spot.shapeData.rectangularSpot.width);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Core height")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(worldSize.y)
                        .defaultValue(origSpot.shapeData.rectangularSpot.height)
                        .format("%.1f"),
                    spot.shapeData.rectangularSpot.height);
            }

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Fade-out radius")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(maxRadius)
                    .defaultValue(origSpot.fadeoutRadius)
                    .format("%.1f"),
                spot.fadeoutRadius);
            ImGui::TreePop();
        }

        /**
         * Flow
         */
        if (ImGui::TreeNodeEx("Flow", flags)) {
            AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Type").values({"None", "Radial flow"}).textWidth(RightColumnWidth).defaultValue(origSpot.flowType),
                spot.flowType);

            if (spot.flowType == FlowType_Radial) {
                AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Orientation")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origSpot.flowData.radialFlow.orientation)
                        .values({"Clockwise", "Counter clockwise"}),
                    spot.flowData.radialFlow.orientation);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Strength")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.5f)
                        .logarithmic(true)
                        .format("%.5f")
                        .defaultValue(origSpot.flowData.radialFlow.strength),
                    spot.flowData.radialFlow.strength);
            }
            ImGui::TreePop();
        }

        /**
         * General physics
         */
        if (ImGui::TreeNodeEx("Physics: General", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Friction")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1)
                    .logarithmic(true)
                    .defaultValue(origSpot.values.friction)
                    .format("%.4f"),
                spot.values.friction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Rigidity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1)
                    .defaultValue(origSpot.values.rigidity)
                    .format("%.2f"),
                spot.values.rigidity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum force")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .defaultValue(origSpot.values.cellMaxForce),
                spot.values.cellMaxForce);
            ImGui::TreePop();
        }

        /**
         * Physics: Radiation
         */
        if (ImGui::TreeNodeEx("Physics: Radiation", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radiation strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .defaultValue(origSpot.values.radiationFactor)
                    .format("%.6f"),
                spot.values.radiationFactor);
            ImGui::TreePop();
        }

        /**
         * Physics: Particle transformation
         */
        if (ImGui::TreeNodeEx("Physics: Particle transformation", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum energy")
                    .textWidth(RightColumnWidth)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSpot.values.cellMinEnergy),
                spot.values.cellMinEnergy);
            ImGui::TreePop();
        }

        /**
         * Collision and binding
         */
        if (ImGui::TreeNodeEx("Physics: Collision and binding", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Binding creation velocity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSpot.values.cellFusionVelocity),
                spot.values.cellFusionVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Binding max energy")
                    .textWidth(RightColumnWidth)
                    .min(50.0f)
                    .max(1000000.0f)
                    .logarithmic(true)
                    .format("%.0f")
                    .defaultValue(origSpot.values.cellMaxBindingEnergy),
                spot.values.cellMaxBindingEnergy);
            if (spot.values.cellMaxBindingEnergy < spot.values.cellMinEnergy + 10.0f) {
                spot.values.cellMaxBindingEnergy = spot.values.cellMinEnergy + 10.0f;
            }
            ImGui::TreePop();
        }

        /**
         * Cell color transition rules
         */
        if (ImGui::TreeNodeEx("Cell color transition rules", flags)) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                ImGui::PushID(color);
                auto parameters = AlienImGui::InputColorTransitionParameters()
                                      .textWidth(RightColumnWidth)
                                      .color(color)
                                      .defaultTargetColor(origSpot.values.cellColorTransitionTargetColor[color])
                                      .defaultTransitionAge(origSpot.values.cellColorTransitionDuration[color])
                                      .logarithmic(true);
                if (0 == color) {
                    parameters.name("Target color and duration");
                }
                AlienImGui::InputColorTransition(
                    parameters, color, spot.values.cellColorTransitionTargetColor[color], spot.values.cellColorTransitionDuration[color]);
                ImGui::PopID();
            }
            ImGui::TreePop();
        }

        /**
         * Mutation 
         */
        if (ImGui::TreeNodeEx("Cell function: Genome mutation probabilities", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neuron weights and bias")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationNeuronDataProbability),
                spot.values.cellFunctionConstructorMutationNeuronDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationDataProbability),
                spot.values.cellFunctionConstructorMutationDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationCellFunctionProbability),
                spot.values.cellFunctionConstructorMutationCellFunctionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationInsertionProbability),
                spot.values.cellFunctionConstructorMutationInsertionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.01f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationDeletionProbability),
                spot.values.cellFunctionConstructorMutationDeletionProbability);
            ImGui::TreePop();
        }

        /**
         * Attacker
         */
        if (ImGui::TreeNodeEx("Cell function: Attacker", flags)) {
            AlienImGui::InputColorMatrix(
                AlienImGui::InputColorMatrixParameters()
                    .name("Food chain color matrix")
                    .textWidth(RightColumnWidth)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerFoodChainColorMatrix)),
                spot.values.cellFunctionAttackerFoodChainColorMatrix);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy cost")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSpot.values.cellFunctionAttackerEnergyCost),
                spot.values.cellFunctionAttackerEnergyCost);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry penalty")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSpot.values.cellFunctionAttackerGeometryDeviationExponent),
                spot.values.cellFunctionAttackerGeometryDeviationExponent);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connections mismatch penalty")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSpot.values.cellFunctionAttackerConnectionsMismatchPenalty),
                spot.values.cellFunctionAttackerConnectionsMismatchPenalty);
        }
        ImGui::TreePop();
    }
    ImGui::EndChild();
    validationAndCorrection(spot);
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j]));
        }
        parameters.radiationAbsorptionByCellColor[i] = std::max(0.0f, std::min(1.0f, parameters.radiationAbsorptionByCellColor[i]));
    }
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParametersSpot& spot) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
        }
    }
}
