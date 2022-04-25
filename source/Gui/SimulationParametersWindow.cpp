#include "SimulationParametersWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "SimulationParametersChanger.h"

namespace
{
    auto const MaxContentTextWidth = 260.0f;
}

_SimulationParametersWindow::_SimulationParametersWindow(SimulationController const& simController)
    : _AlienWindow("Simulation parameters", "windows.simulation parameters", false)
    , _simController(simController)
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
    auto simParameters = _simController->getSimulationParameters();
    auto origSimParameters = _simController->getOriginalSimulationParameters();
    auto lastSimParameters = simParameters;

    auto simParametersSpots = _simController->getSimulationParametersSpots();
    auto origSimParametersSpots = _simController->getOriginalSimulationParametersSpots();
    auto lastSimParametersSpots = simParametersSpots;

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(78)), false)) {

        if (ImGui::BeginTabBar("##Flow", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

            if (simParametersSpots.numSpots < 2) {
                if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                    int index = simParametersSpots.numSpots;
                    simParametersSpots.spots[index] = createSpot(simParameters, index);
                    _simController->setOriginalSimulationParametersSpot(simParametersSpots.spots[index], index);
                    ++simParametersSpots.numSpots;
                }
                AlienImGui::Tooltip("Add spot");
            }

            if (ImGui::BeginTabItem("Base", NULL, ImGuiTabItemFlags_None)) {
                processBase(simParameters, origSimParameters);
                ImGui::EndTabItem();
            }

            for (int tab = 0; tab < simParametersSpots.numSpots; ++tab) {
                SimulationParametersSpot& spot = simParametersSpots.spots[tab];
                SimulationParametersSpot const& origSpot = origSimParametersSpots.spots[tab];
                bool open = true;
                std::string name = "Spot " + std::to_string(tab+1);
                if (ImGui::BeginTabItem(name.c_str(), &open, ImGuiTabItemFlags_None)) {
                    processSpot(spot, origSpot);
                    ImGui::EndTabItem();
                }

                if (!open) {
                    for (int i = tab; i < simParametersSpots.numSpots - 1; ++i) {
                        simParametersSpots.spots[i] = simParametersSpots.spots[i + 1];
                        _simController->setOriginalSimulationParametersSpot(simParametersSpots.spots[i], i);
                    }
                    --simParametersSpots.numSpots;
                }
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    if (AlienImGui::ToggleButton("Change automatically", _changeAutomatically)) {
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
    if(AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Epoch time steps")
                .defaultValue(_simulationParametersChanger->getOriginalTimestepsPerEpoch())
                .textWidth(MaxContentTextWidth)
                .tooltip("Duration in time steps after which a change is applied."),
        timestepsPerEpoch)) {
        _simulationParametersChanger->setTimestepsPerEpoch(timestepsPerEpoch);
    }
    ImGui::EndDisabled();

    if (simParameters != lastSimParameters) {
        _simController->setSimulationParameters_async(simParameters);
    }

    if (simParametersSpots != lastSimParametersSpots) {
        _simController->setSimulationParametersSpots_async(simParametersSpots);
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
    spot.coreRadius = maxRadius / 3;
    spot.fadeoutRadius = maxRadius / 3;
    spot.color = _savedPalette[(2 + index) * 8];

    spot.values = simParameters.spotValues;
    return spot;
}

void _SimulationParametersWindow::processBase(
    SimulationParameters& simParameters,
    SimulationParameters const& origSimParameters)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        /**
         * Numerics
         */
        AlienImGui::Group("Numerics");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Time step size")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSimParameters.timestepSize)
                .tooltip(std::string("Time duration calculated in a single step. Smaller values increase the accuracy "
                                     "of the simulation.")),
            simParameters.timestepSize);

        /**
         * General physics
         */
        AlienImGui::Group("General physics");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Friction")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .logarithmic(true)
                .format("%.4f")
                .defaultValue(origSimParameters.spotValues.friction)
                .tooltip(std::string("Specifies how much the movements are slowed down per time step.")),
            simParameters.spotValues.friction);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Rigidity")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(origSimParameters.spotValues.rigidity)
                .tooltip(std::string("Controls the rigidity of connected cells.\nA higher value will cause connected cells to move more uniformly.")),
            simParameters.spotValues.rigidity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation strength")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.01f)
                .logarithmic(true)
                .format("%.5f")
                .defaultValue(origSimParameters.spotValues.radiationFactor)
                .tooltip(std::string("Indicates how energetic the emitted particles of cells are.")),
            simParameters.spotValues.radiationFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum velocity")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(6.0f)
                .defaultValue(origSimParameters.cellMaxVel)
                .tooltip(std::string("Maximum velocity that a cell can reach.")),
            simParameters.cellMaxVel);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum force")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(3.0f)
                .defaultValue(origSimParameters.spotValues.cellMaxForce)
                .tooltip(std::string("Maximum force that can be applied to a cell without causing it to disintegrate.")),
            simParameters.spotValues.cellMaxForce);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum energy")
                .textWidth(MaxContentTextWidth)
                .min(10.0f)
                .max(200.0f)
                .defaultValue(origSimParameters.spotValues.cellMinEnergy)
                .tooltip(std::string("Minimum energy a cell needs to exist.")),
            simParameters.spotValues.cellMinEnergy);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum distance")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSimParameters.cellMinDistance)
                .tooltip(std::string("Minimum distance between two cells without them annihilating each other.")),
            simParameters.cellMinDistance);

        /**
         * Collision and binding
         */
        AlienImGui::Group("Collision and binding");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Repulsion strength")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.3f)
                .defaultValue(origSimParameters.cellRepulsionStrength)
                .tooltip(std::string("The strength of the repulsive forces, between two cells that do not connect.")),
            simParameters.cellRepulsionStrength);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum collision distance")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(3.0f)
                .defaultValue(origSimParameters.cellMaxCollisionDistance)
                .tooltip(std::string("Maximum distance up to which a collision of two cells is possible.")),
            simParameters.cellMaxCollisionDistance);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum binding distance")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(5.0f)
                .defaultValue(origSimParameters.cellMaxBindingDistance)
                .tooltip(std::string("Maximum distance up to which a connection of two cells is possible.")),
            simParameters.cellMaxBindingDistance);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding force strength")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(4.0f)
                .defaultValue(origSimParameters.spotValues.cellBindingForce)
                .tooltip(std::string(
                    "Strength of the force that holds two connected cells together. For larger binding forces, the "
                    "time step size should be selected smaller due to numerical instabilities.")),
            simParameters.spotValues.cellBindingForce);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding creation velocity")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSimParameters.spotValues.cellFusionVelocity)
                .tooltip(
                    std::string("Minimum velocity of two colliding cells so that a connection can be established.")),
            simParameters.spotValues.cellFusionVelocity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding maximum energy")
                .textWidth(MaxContentTextWidth)
                .min(50.0f)
                .max(1000000.0f)
                .logarithmic(true)
                .format("%.0f")
                .defaultValue(origSimParameters.spotValues.cellMaxBindingEnergy)
                .tooltip(std::string("Maximum energy of a cell at which they can maintain a connection.")),
            simParameters.spotValues.cellMaxBindingEnergy);
        if (simParameters.spotValues.cellMaxBindingEnergy < simParameters.spotValues.cellMinEnergy + 10.0f) {
            simParameters.spotValues.cellMaxBindingEnergy = simParameters.spotValues.cellMinEnergy + 10.0f;
        }
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Maximum cell bonds")
                .textWidth(MaxContentTextWidth)
                .defaultValue(origSimParameters.cellMaxBonds)
                .min(0)
                .max(6)
                .tooltip(std::string("Maximum number of connections a cell can establish with others.")),
            simParameters.cellMaxBonds);

        /**
         * Mutations
         */
        AlienImGui::Group("Mutations");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell mutation rate")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.001f)
                .logarithmic(true)
                .format("%.7f")
                .defaultValue(origSimParameters.spotValues.cellMutationRate)
                .tooltip(std::string("Probability that a byte or property of a cell is changed per time step.")),
            simParameters.spotValues.cellMutationRate);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Token mutation rate")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.1f)
                .logarithmic(true)
                .format("%.5f")
                .defaultValue(origSimParameters.spotValues.tokenMutationRate)
                .tooltip(std::string("Probability that a memory byte of a token is changed per time step.")),
            simParameters.spotValues.tokenMutationRate);

        /**
         * Cell specialization: General
         */
        AlienImGui::Group("Cell specialization: General");
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Minimum invocations")
                .textWidth(MaxContentTextWidth)
                .defaultValue(origSimParameters.spotValues.cellFunctionMinInvocations)
                .min(0)
                .max(1000000)
                .logarithmic(true)
                .tooltip(std::string("Minimum number of invocations of a cell function up to which the cell remains stable. If the value is exceeded, the cell "
                                     "decays with a certain probability.")),
            simParameters.spotValues.cellFunctionMinInvocations);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Invocation decay probability")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .logarithmic(true)
                .format("%.6f")
                .defaultValue(origSimParameters.spotValues.cellFunctionInvocationDecayProb)
                .tooltip(std::string("The probability that a cell will be destroyed after its function is invoked more times than the minimum invocations "
                                     "parameter specifies.")),
            simParameters.spotValues.cellFunctionInvocationDecayProb);

        /**
         * Cell specialization: Digestion function
         */
        AlienImGui::Group("Cell specialization: Digestion function");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(4.0f)
                .defaultValue(origSimParameters.spotValues.cellFunctionWeaponEnergyCost)
                .tooltip(std::string(
                    "Amount of energy lost by an attempted attack of a cell in the form of emitted energy particles.")),
            simParameters.spotValues.cellFunctionWeaponEnergyCost);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Target color mismatch penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.1f)
                .defaultValue(origSimParameters.spotValues.cellFunctionWeaponColorTargetMismatchPenalty)
                .tooltip(std::string("The larger this value is, the less energy a cell can gain from an attack if the attacked cell "
                                     "does not match the target color. ATTENTION: A value above 1 means that a mismatched color is treated as poisoned.")),
            simParameters.spotValues.cellFunctionWeaponColorTargetMismatchPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Successive color dominance")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSimParameters.spotValues.cellFunctionWeaponColorDominance)
                .tooltip(std::string("The larger this value is, the less energy a cell can gain from an attack if the attacked cell does not have the successive "
                                     "color of the attacker cell.")),
            simParameters.spotValues.cellFunctionWeaponColorDominance);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(5.0f)
                .defaultValue(origSimParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent)
                .tooltip(
                    std::string("The larger this value is, the less energy a cell can gain from an attack if the local "
                                "geometry of the attacked cell does not match the attacking cell.")),
            simParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connections mismatch penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSimParameters.spotValues.cellFunctionWeaponConnectionsMismatchPenalty)
                .tooltip(std::string("The larger this parameter is, the more difficult it is to digest cells that contain more connections.")),
            simParameters.spotValues.cellFunctionWeaponConnectionsMismatchPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Token penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSimParameters.spotValues.cellFunctionWeaponTokenPenalty)
                .tooltip(std::string("The larger this parameter is, the more difficult it is to digest cells that contains a token.")),
            simParameters.spotValues.cellFunctionWeaponTokenPenalty);

        /**
         * Cell specialization: Sensor function
         */
        AlienImGui::Group("Cell specialization: Sensor function");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Range")
                .textWidth(MaxContentTextWidth)
                .min(10.0f)
                .max(512.0f)
                .defaultValue(origSimParameters.cellFunctionSensorRange)
                .tooltip(std::string("The maximum radius in which a sensor can detect mass concentrations.")),
            simParameters.cellFunctionSensorRange);
    }
    ImGui::EndChild();
}

void _SimulationParametersWindow::processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto worldSize = _simController->getWorldSize();

        /**
         * Location and metadata
         */
        AlienImGui::Group("Location and metadata");

        auto& color = spot.color;
        AlienImGui::ColorButtonWithPicker(
            "##",
            color,
            _backupColor,
            _savedPalette,
            RealVector2D(
                ImGui::GetContentRegionAvail().x - StyleRepository::getInstance().scaleContent(MaxContentTextWidth),
                0));

        ImGui::SameLine();
        ImGui::Text("Background color");

        int shape = static_cast<int>(spot.shape);
        AlienImGui::Combo(
            AlienImGui::ComboParameters()
                .name("Shape")
                .values({"Circular", "Rectangular"})
                .textWidth(MaxContentTextWidth)
                .defaultValue(static_cast<int>(origSpot.shape)),
            shape);
        spot.shape = static_cast<SpotShape>(shape);
        auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Position X")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(toFloat(worldSize.x))
                .defaultValue(origSpot.posX)
                .format("%.1f"),
            spot.posX);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Position Y")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(toFloat(worldSize.y))
                .defaultValue(origSpot.posY)
                .format("%.1f"),
            spot.posY);
        if (spot.shape == SpotShape::Circular) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Core radius")
                    .textWidth(MaxContentTextWidth)
                    .min(0)
                    .max(maxRadius)
                    .defaultValue(origSpot.coreRadius)
                    .format("%.1f"),
                spot.coreRadius);
        }
        if (spot.shape == SpotShape::Rectangular) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Core width")
                    .textWidth(MaxContentTextWidth)
                    .min(0)
                    .max(worldSize.x)
                    .defaultValue(origSpot.width)
                    .format("%.1f"),
                spot.width);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Core height")
                    .textWidth(MaxContentTextWidth)
                    .min(0)
                    .max(worldSize.y)
                    .defaultValue(origSpot.height)
                    .format("%.1f"),
                spot.height);
        }

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Fade-out radius")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(maxRadius)
                .defaultValue(origSpot.fadeoutRadius)
                .format("%.1f"),
            spot.fadeoutRadius);

        /**
         * General physics
         */
        AlienImGui::Group("General physics");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Friction")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1)
                .logarithmic(true)
                .defaultValue(origSpot.values.friction)
                .format("%.4f"),
            spot.values.friction);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Rigidity")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1)
                .defaultValue(origSpot.values.rigidity)
                .format("%.2f"),
            spot.values.rigidity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation strength")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.01f)
                .logarithmic(true)
                .defaultValue(origSpot.values.radiationFactor)
                .format("%.5f"),
            spot.values.radiationFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum force")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(3.0f)
                .defaultValue(origSpot.values.cellMaxForce),
            spot.values.cellMaxForce);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum energy")
                .textWidth(MaxContentTextWidth)
                .min(10.0f)
                .max(200.0f)
                .defaultValue(origSpot.values.cellMinEnergy),
            spot.values.cellMinEnergy);

        /**
         * Collision and binding
         */
        AlienImGui::Group("Collision and binding");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding force strength")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(4.0f)
                .defaultValue(origSpot.values.cellBindingForce),
            spot.values.cellBindingForce);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding creation velocity")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSpot.values.cellFusionVelocity),
            spot.values.cellFusionVelocity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding max energy")
                .textWidth(MaxContentTextWidth)
                .min(50.0f)
                .max(1000000.0f)
                .logarithmic(true)
                .format("%.0f")
                .defaultValue(origSpot.values.cellMaxBindingEnergy),
            spot.values.cellMaxBindingEnergy);
        if (spot.values.cellMaxBindingEnergy < spot.values.cellMinEnergy + 10.0f) {
            spot.values.cellMaxBindingEnergy = spot.values.cellMinEnergy + 10.0f;
        }

        /**
         * Mutations
         */
        AlienImGui::Group("Mutations");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell mutation rate")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.001f)
                .logarithmic(true)
                .format("%.7f")
                .defaultValue(origSpot.values.cellMutationRate),
            spot.values.cellMutationRate);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Token mutation rate")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(0.1f)
                .logarithmic(true)
                .format("%.5f")
                .defaultValue(origSpot.values.tokenMutationRate),
            spot.values.tokenMutationRate);

        /**
         * Cell specialization: General
         */
        AlienImGui::Group("Cell specialization: General");
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Minimum invocations")
                .textWidth(MaxContentTextWidth)
                .defaultValue(origSpot.values.cellFunctionMinInvocations)
                .min(0)
                .max(1000000)
                .logarithmic(true),
            spot.values.cellFunctionMinInvocations);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Invocation decay probability")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .logarithmic(true)
                .format("%.6f")
                .defaultValue(origSpot.values.cellFunctionInvocationDecayProb),
            spot.values.cellFunctionInvocationDecayProb);

        /**
         * Cell specialization: Digestion function
         */
        AlienImGui::Group("Cell specialization: Digestion function");
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(4.0f)
                .defaultValue(origSpot.values.cellFunctionWeaponEnergyCost),
            spot.values.cellFunctionWeaponEnergyCost);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Target color mismatch penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.1f)
                .defaultValue(origSpot.values.cellFunctionWeaponColorTargetMismatchPenalty),
            spot.values.cellFunctionWeaponColorTargetMismatchPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Successive color dominance")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSpot.values.cellFunctionWeaponColorDominance),
            spot.values.cellFunctionWeaponColorDominance);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(5.0f)
                .defaultValue(origSpot.values.cellFunctionWeaponGeometryDeviationExponent),
            spot.values.cellFunctionWeaponGeometryDeviationExponent);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connections mismatch penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSpot.values.cellFunctionWeaponConnectionsMismatchPenalty),
            spot.values.cellFunctionWeaponConnectionsMismatchPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Token penalty")
                .textWidth(MaxContentTextWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(origSpot.values.cellFunctionWeaponTokenPenalty),
            spot.values.cellFunctionWeaponTokenPenalty);
    }
    ImGui::EndChild();
}

