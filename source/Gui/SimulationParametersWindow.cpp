#include "SimulationParametersWindow.h"

#include "imgui.h"

#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "EngineImpl/SimulationController.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"

_SimulationParametersWindow::_SimulationParametersWindow(
    StyleRepository const& styleRepository,
    SimulationController const& simController)
    : _styleRepository(styleRepository)
    , _simController(simController)
{
    for (int n = 0; n < IM_ARRAYSIZE(_savedPalette); n++) {
        ImVec4 color;
        ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.2f, color.x, color.y, color.z);
        color.w = 1.0f; //alpha
        _savedPalette[n] = static_cast<ImU32>(ImColor(color));
    }
    _on = GlobalSettings::getInstance().getBoolState("windows.simulation parameters.active", false);
}

_SimulationParametersWindow::~_SimulationParametersWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.simulation parameters.active", _on);
}

void _SimulationParametersWindow::process()
{
    if (!_on) {
        return;
    }
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    auto simParameters = _simController->getSimulationParameters();
    auto origSimParameters = _simController->getOriginalSimulationParameters();
    auto lastSimParameters = simParameters;

    auto simParametersSpots = _simController->getSimulationParametersSpots();
    auto origSimParametersSpots = _simController->getOriginalSimulationParametersSpots();
    auto lastSimParametersSpots = simParametersSpots;

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Simulation parameters", &_on, windowFlags)) {

        if (ImGui::BeginTabBar(
                "##Flow", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

            if (simParametersSpots.numSpots < 2) {
                if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                    int index = simParametersSpots.numSpots;
                    simParametersSpots.spots[index] = createSpot(simParameters, index);
                    _simController->setOriginalSimulationParametersSpot(simParametersSpots.spots[index], index);
                    ++simParametersSpots.numSpots;
                }
            }

            if (ImGui::BeginTabItem("Base", NULL, ImGuiTabItemFlags_None)) {
                processBase(simParameters, origSimParameters);
                ImGui::EndTabItem();
            }

            for (int tab = 0; tab < simParametersSpots.numSpots; ++tab) {
                SimulationParametersSpot& spot = simParametersSpots.spots[tab];
                SimulationParametersSpot const& origSpot = origSimParametersSpots.spots[tab];
                bool open = true;
                char name[16];
                snprintf(name, IM_ARRAYSIZE(name), "Spot %01d", tab + 1);
                if (ImGui::BeginTabItem(name, &open, ImGuiTabItemFlags_None)) {
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

        ImGui::End();
    }

    if (simParameters != lastSimParameters) {
        _simController->setSimulationParameters_async(simParameters);
    }

    if (simParametersSpots != lastSimParametersSpots) {
        _simController->setSimulationParametersSpots_async(simParametersSpots);
    }
}

bool _SimulationParametersWindow::isOn() const
{
    return _on;
}

void _SimulationParametersWindow::setOn(bool value)
{
    _on = value;
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
        createGroup("Numerics");
        AlienImGui::SliderFloat(
            "Time step size",
            simParameters.timestepSize,
            origSimParameters.timestepSize,
            0,
            1.0f,
            false,
            "%.3f",
            std::string("Time duration calculated in a single step. Smaller values increase the accuracy of the simulation."));

        createGroup("General physics");
        AlienImGui::SliderFloat(
            "Friction",
            simParameters.spotValues.friction,
            origSimParameters.spotValues.friction,
            0,
            1.0f,
            true,
            "%.4f",
            std::string("Specifies how much the movements are slowed down per time step."));
        AlienImGui::SliderFloat(
            "Radiation strength",
            simParameters.spotValues.radiationFactor,
            origSimParameters.spotValues.radiationFactor,
            0,
            0.01f,
            true,
            "%.5f",
            std::string("Indicates how energetic the emitted particles of cells are."));
        AlienImGui::SliderFloat(
            "Maximum velocity",
            simParameters.cellMaxVel,
            origSimParameters.cellMaxVel,
            0,
            6.0f,
            false,
            "%.3f",
            std::string("Maximum velocity that a cell can reach."));
        AlienImGui::SliderFloat(
            "Maximum force",
            simParameters.spotValues.cellMaxForce,
            origSimParameters.spotValues.cellMaxForce,
            0,
            3.0f,
            false,
            "%.3f",
            std::string("Maximum force that can be applied to a cell without causing it to disintegrate."));
        AlienImGui::SliderFloat(
            "Minimum energy",
            simParameters.spotValues.cellMinEnergy,
            origSimParameters.spotValues.cellMinEnergy,
            0,
            100.0f,
            false,
            "%.3f",
            std::string("Minimum energy a cell needs to exist."));
        AlienImGui::SliderFloat(
            "Minimum distance",
            simParameters.cellMinDistance,
            origSimParameters.cellMinDistance,
            0,
            1.0f,
            false,
            "%.3f",
            std::string("Minimum distance between two cells without them annihilating each other."));

        createGroup("Collision and binding");
        AlienImGui::SliderFloat(
            "Repulsion strength",
            simParameters.cellRepulsionStrength,
            origSimParameters.cellRepulsionStrength,
            0,
            0.3f,
            false,
            "%.3f",
            std::string("The strength of the repulsive forces, between two cells that do not connect."));
        AlienImGui::SliderFloat(
            "Maximum collision distance",
            simParameters.cellMaxCollisionDistance,
            origSimParameters.cellMaxCollisionDistance,
            0,
            3.0f,
            false,
            "%.3f",
            std::string("Maximum distance up to which a collision of two cells is possible."));
        AlienImGui::SliderFloat(
            "Maximum binding distance",
            simParameters.cellMaxBindingDistance,
            origSimParameters.cellMaxBindingDistance,
            0,
            5.0f,
            false,
            "%.3f",
            std::string("Maximum distance up to which a connection of two cells is possible."));
        AlienImGui::SliderFloat(
            "Binding force strength",
            simParameters.spotValues.cellBindingForce,
            origSimParameters.spotValues.cellBindingForce,
            0,
            4.0f,
            false,
            "%.3f",
            std::string("Strength of the force that holds two connected cells together. For larger binding forces, the "
                        "time step size should be selected smaller due to numerical instabilities."));
        AlienImGui::SliderFloat(
            "Binding creation force",
            simParameters.spotValues.cellFusionVelocity,
            origSimParameters.spotValues.cellFusionVelocity,
            0,
            1.0f,
            false,
            "%.3f",
            std::string("Minimum collision velocity of two cells so that a connection can be established."));
        AlienImGui::SliderInt(
            "Maximum cell bonds",
            simParameters.cellMaxBonds,
            origSimParameters.cellMaxBonds,
            0,
            6,
            std::string("Maximum number of connections a cell can establish with others."));

        createGroup("Cell functions");
        AlienImGui::SliderFloat(
            "Mutation rate",
            simParameters.spotValues.tokenMutationRate,
            origSimParameters.spotValues.tokenMutationRate,
            0,
            0.005f,
            false,
            "%.5f",
            std::string("Probability that a byte in the token memory is changed per time step."));
        AlienImGui::SliderFloat(
            "Weapon energy cost",
            simParameters.spotValues.cellFunctionWeaponEnergyCost,
            origSimParameters.spotValues.cellFunctionWeaponEnergyCost,
            0,
            4.0f,
            false,
            "%.3f",
            std::string("Amount of energy lost by an attempted attack of a cell in the form of emitted energy particles."));
        AlienImGui::SliderFloat(
            "Weapon color penalty",
            simParameters.spotValues.cellFunctionWeaponColorPenalty,
            origSimParameters.spotValues.cellFunctionWeaponColorPenalty,
            0,
            1.0f,
            false,
            "%.3f",
            std::string("The larger this value is, the less energy a cell can gain from an attack if the attacked cell "
                        "does not match the adjacent color."));
        AlienImGui::SliderFloat(
            "Weapon geometric penalty",
            simParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent,
            origSimParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent,
            0,
            5.0f,
            false,
            "%.3f",
            std::string("The larger this value is, the less energy a cell can gain from an attack if the local "
                        "geometry of the attacked cell does not match the attacking cell."));
        ImGui::EndChild();
    }
}

void _SimulationParametersWindow::processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto worldSize = _simController->getWorldSize();

        createGroup("Location and meta data");

        auto& color = spot.color;
        AlienImGui::ColorButtonWithPicker(
            "##", color, _backupColor, _savedPalette, RealVector2D(ImGui::GetContentRegionAvail().x / 2, 0));

        ImGui::SameLine();
        ImGui::Text("Background color");

        auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
        AlienImGui::SliderFloat(" Position X", spot.posX, origSpot.posX, 0, toFloat(worldSize.x), false, "%.1f");
        AlienImGui::SliderFloat(" Position Y", spot.posY, origSpot.posY, 0, toFloat(worldSize.y), false, "%.1f");
        AlienImGui::SliderFloat(" Core radius", spot.coreRadius, origSpot.coreRadius, 0, maxRadius, false, "%.1f");
        AlienImGui::SliderFloat(" Fade-out radius", spot.fadeoutRadius, origSpot.fadeoutRadius, 0, maxRadius, false, "%.1f");

        createGroup("General physics");
        AlienImGui::SliderFloat("Friction", spot.values.friction, origSpot.values.friction, 0, 1.0f, true, "%.4f");
        AlienImGui::SliderFloat(
            "Radiation strength", spot.values.radiationFactor, origSpot.values.radiationFactor, 0, 0.01f, true, "%.5f");
        AlienImGui::SliderFloat("Maximum force", spot.values.cellMaxForce, origSpot.values.cellMaxForce, 0, 3.0f);
        AlienImGui::SliderFloat("Minimum energy", spot.values.cellMinEnergy, origSpot.values.cellMinEnergy, 0, 100.0f);

        createGroup("Collision and binding");
        AlienImGui::SliderFloat(
            "Binding force strength", spot.values.cellBindingForce, origSpot.values.cellBindingForce, 0, 4.0f);
        AlienImGui::SliderFloat(
            "Binding creation force", spot.values.cellFusionVelocity, origSpot.values.cellFusionVelocity, 0, 1.0f);

        createGroup("Cell functions");
        AlienImGui::SliderFloat(
            "Mutation rate",
            spot.values.tokenMutationRate,
            origSpot.values.tokenMutationRate,
            0,
            0.005f,
            false,
            "%.5f");
        AlienImGui::SliderFloat(
            "Weapon energy cost",
            spot.values.cellFunctionWeaponEnergyCost,
            origSpot.values.cellFunctionWeaponEnergyCost,
            0,
            4.0f);
        AlienImGui::SliderFloat(
            "Weapon color penalty",
            spot.values.cellFunctionWeaponColorPenalty,
            origSpot.values.cellFunctionWeaponColorPenalty,
            0,
            1.0f);
        AlienImGui::SliderFloat(
            "Weapon geometric penalty",
            spot.values.cellFunctionWeaponGeometryDeviationExponent,
            origSpot.values.cellFunctionWeaponGeometryDeviationExponent,
            0,
            5.0f);

        ImGui::EndChild();
    }
}

void _SimulationParametersWindow::createGroup(std::string const& name)
{
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(name.c_str());
    ImGui::Separator();
    ImGui::Spacing();
}

