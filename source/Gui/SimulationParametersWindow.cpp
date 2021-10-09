#include "SimulationParametersWindow.h"

#include "imgui.h"

#include "EngineImpl/SimulationController.h"

#include "Widgets.h"
#include "StyleRepository.h"

_SimulationParametersWindow::_SimulationParametersWindow(
    StyleRepository const& styleRepository,
    SimulationController const& simController)
    : _styleRepository(styleRepository)
    , _simController(simController)
{}

void _SimulationParametersWindow::process()
{
    if (!_on) {
        return;
    }
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    auto simParameters = _simController->getSimulationParameters();
    auto origSimParameters = simParameters;

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Simulation parameters", &_on, windowFlags);

    createGroup("Numerics");
    createFloatItem("Time step size", simParameters.timestepSize, 0, 1.0f);

    createGroup("General physics");
    createFloatItem("Friction", simParameters.friction, 0, 1.0f, true, "%.4f");
    createFloatItem("Radiation strength", simParameters.radiationFactor, 0, 0.01f, true, "%.5f");
    createFloatItem("Maximum velocity", simParameters.cellMaxVel, 0, 6.0f);
    createFloatItem("Maximum force", simParameters.cellMaxForce, 0, 3.0f);
    createFloatItem("Minimum energy", simParameters.cellMinEnergy, 0, 100.0f);
    createFloatItem("Minimum distance", simParameters.cellMinDistance, 0, 1.0f);

    createGroup("Collision and binding");
    createFloatItem("Repulsion strength", simParameters.repulsionStrength, 0, 0.3f);
    createFloatItem("Maximum collision distance", simParameters.cellMaxCollisionDistance, 0, 3.0f);
    createFloatItem("Maximum binding distance", simParameters.cellMaxBindingDistance, 0, 5.0f);
    createFloatItem("Binding force strength", simParameters.bindingForce, 0, 4.0f);
    createFloatItem("Binding creation force", simParameters.cellFusionVelocity, 0, 1.0f);
    createIntItem("Maximum cell bonds", simParameters.cellMaxBonds, 0, 6);

    createGroup("Cell functions");
    createFloatItem("Mutation rate", simParameters.tokenMutationRate, 0, 0.005f, false, "%.5f");
    createFloatItem("Weapon energy cost", simParameters.cellFunctionWeaponEnergyCost, 0, 4.0f);
    auto weaponColorPenalty = 1.0f - simParameters.cellFunctionWeaponInhomogeneousColorFactor;
    createFloatItem("Weapon color penalty", weaponColorPenalty, 0, 1.0f);
    simParameters.cellFunctionWeaponInhomogeneousColorFactor = 1.0f - weaponColorPenalty;
    createFloatItem("Weapon geometric penalty", simParameters.cellFunctionWeaponGeometryDeviationExponent, 0, 5.0f);

    ImGui::End();

    if (simParameters != origSimParameters) {
        _simController->setSimulationParameters_async(simParameters);
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
    float min,
    float max,
    bool logarithmic,
    std::string const& format)
{ 
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x/2);
    ImGui::SliderFloat(name.c_str(), &value, min, max, format.c_str(), logarithmic ? ImGuiSliderFlags_Logarithmic : 0);

    Widgets::processHelpMarker("This is a more typical looking tree with selectable nodes.\n"
               "Click to select, CTRL+Click to toggle, click on arrows or double-click to open.");
    ImGui::Spacing();
}

void _SimulationParametersWindow::createIntItem(std::string const& name, int& value, int min, int max)
{
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x/2);
    ImGui::SliderInt(name.c_str(), &value, min, max);

    Widgets::processHelpMarker("This is a more typical looking tree with selectable nodes.\n"
               "Click to select, CTRL+Click to toggle, click on arrows or double-click to open.");
    ImGui::Spacing();
}
