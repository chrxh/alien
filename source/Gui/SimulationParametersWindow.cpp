#include "SimulationParametersWindow.h"

#include "imgui.h"

#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "EngineImpl/SimulationController.h"

#include "AlienImGui.h"
#include "StyleRepository.h"

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

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Simulation parameters", &_on, windowFlags);

    if (ImGui::BeginTabBar("##Flow", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

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
        createFloatItem("Time step size", simParameters.timestepSize, origSimParameters.timestepSize, 0, 1.0f);

        createGroup("General physics");
        createFloatItem(
            "Friction",
            simParameters.spotValues.friction,
            origSimParameters.spotValues.friction,
            0,
            1.0f,
            true,
            "%.4f");
        createFloatItem(
            "Radiation strength",
            simParameters.spotValues.radiationFactor,
            origSimParameters.spotValues.radiationFactor,
            0,
            0.01f,
            true,
            "%.5f");
        createFloatItem("Maximum velocity", simParameters.cellMaxVel, origSimParameters.cellMaxVel, 0, 6.0f);
        createFloatItem(
            "Maximum force", simParameters.spotValues.cellMaxForce, origSimParameters.spotValues.cellMaxForce, 0, 3.0f);
        createFloatItem(
            "Minimum energy",
            simParameters.spotValues.cellMinEnergy,
            origSimParameters.spotValues.cellMinEnergy,
            0,
            100.0f);
        createFloatItem("Minimum distance", simParameters.cellMinDistance, origSimParameters.cellMinDistance, 0, 1.0f);

        createGroup("Collision and binding");
        createFloatItem(
            "Repulsion strength",
            simParameters.cellRepulsionStrength,
            origSimParameters.cellRepulsionStrength,
            0,
            0.3f);
        createFloatItem(
            "Maximum collision distance",
            simParameters.cellMaxCollisionDistance,
            origSimParameters.cellMaxCollisionDistance,
            0,
            3.0f);
        createFloatItem(
            "Maximum binding distance",
            simParameters.cellMaxBindingDistance,
            origSimParameters.cellMaxBindingDistance,
            0,
            5.0f);
        createFloatItem(
            "Binding force strength",
            simParameters.spotValues.cellBindingForce,
            origSimParameters.spotValues.cellBindingForce,
            0,
            4.0f);
        createFloatItem(
            "Binding creation force",
            simParameters.spotValues.cellFusionVelocity,
            origSimParameters.spotValues.cellFusionVelocity,
            0,
            1.0f);
        createIntItem("Maximum cell bonds", simParameters.cellMaxBonds, origSimParameters.cellMaxBonds, 0, 6);

        createGroup("Cell functions");
        createFloatItem(
            "Mutation rate",
            simParameters.spotValues.tokenMutationRate,
            origSimParameters.spotValues.tokenMutationRate,
            0,
            0.005f,
            false,
            "%.5f");
        createFloatItem(
            "Weapon energy cost",
            simParameters.spotValues.cellFunctionWeaponEnergyCost,
            origSimParameters.spotValues.cellFunctionWeaponEnergyCost,
            0,
            4.0f);
        createFloatItem(
            "Weapon color penalty",
            simParameters.spotValues.cellFunctionWeaponColorPenalty,
            origSimParameters.spotValues.cellFunctionWeaponColorPenalty,
            0,
            1.0f);
        createFloatItem(
            "Weapon geometric penalty",
            simParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent,
            origSimParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent,
            0,
            5.0f);
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
        createFloatItem(" Position X", spot.posX, origSpot.posX, 0, toFloat(worldSize.x), false, "%.1f");
        createFloatItem(" Position Y", spot.posY, origSpot.posY, 0, toFloat(worldSize.y), false, "%.1f");
        createFloatItem(" Core radius", spot.coreRadius, origSpot.coreRadius, 0, maxRadius, false, "%.1f");
        createFloatItem(" Fade-out radius", spot.fadeoutRadius, origSpot.fadeoutRadius, 0, maxRadius, false, "%.1f");

        createGroup("General physics");
        createFloatItem("Friction", spot.values.friction, origSpot.values.friction, 0, 1.0f, true, "%.4f");
        createFloatItem(
            "Radiation strength", spot.values.radiationFactor, origSpot.values.radiationFactor, 0, 0.01f, true, "%.5f");
        createFloatItem("Maximum force", spot.values.cellMaxForce, origSpot.values.cellMaxForce, 0, 3.0f);
        createFloatItem("Minimum energy", spot.values.cellMinEnergy, origSpot.values.cellMinEnergy, 0, 100.0f);

        createGroup("Collision and binding");
        createFloatItem(
            "Binding force strength", spot.values.cellBindingForce, origSpot.values.cellBindingForce, 0, 4.0f);
        createFloatItem(
            "Binding creation force", spot.values.cellFusionVelocity, origSpot.values.cellFusionVelocity, 0, 1.0f);

        createGroup("Cell functions");
        createFloatItem(
            "Mutation rate",
            spot.values.tokenMutationRate,
            origSpot.values.tokenMutationRate,
            0,
            0.005f,
            false,
            "%.5f");
        createFloatItem(
            "Weapon energy cost",
            spot.values.cellFunctionWeaponEnergyCost,
            origSpot.values.cellFunctionWeaponEnergyCost,
            0,
            4.0f);
        createFloatItem(
            "Weapon color penalty",
            spot.values.cellFunctionWeaponColorPenalty,
            origSpot.values.cellFunctionWeaponColorPenalty,
            0,
            1.0f);
        createFloatItem(
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

void _SimulationParametersWindow::createFloatItem(
    std::string const& name,
    float& value,
    float defaultValue,
    float min,
    float max,
    bool logarithmic,
    std::string const& format,
    boost::optional<std::string> help)
{ 
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x/2);
    ImGui::SliderFloat(("##" + name).c_str(), &value, min, max, format.c_str(), logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
    ImGui::SameLine();
    ImGui::BeginDisabled(value == defaultValue);
    if (ImGui::Button((ICON_FA_UNDO "##" + name).c_str())) {
        value = defaultValue;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text(name.c_str());
    if (help) {
        AlienImGui::HelpMarker(help->c_str());
    }
}

void _SimulationParametersWindow::createIntItem(
    std::string const& name,
    int& value,
    int defaultValue,
    int min,
    int max,
    boost::optional<std::string> help)
{
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x/2);
    ImGui::SliderInt(("##" + name).c_str(), &value, min, max);
    ImGui::SameLine();
    ImGui::BeginDisabled(value == defaultValue);
    if (ImGui::Button((ICON_FA_UNDO "##" + name).c_str())) {
        value = defaultValue;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text(name.c_str());

    if (help) {
        AlienImGui::HelpMarker(help->c_str());
    }
}
