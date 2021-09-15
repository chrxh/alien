#include "SimulationParametersWindow.h"

#include "imgui.h"

#include "StyleRepository.h"

_SimulationParametersWindow::_SimulationParametersWindow(StyleRepository const& styleRepository)
    : _styleRepository(styleRepository)
{}

void _SimulationParametersWindow::process()
{
    if (_on) {
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
        ImGui::Begin("Simulation parameters", &_on, windowFlags);
        createGroup("General physics");
        static float friction = 0.5f;
        createFloatItem("Friction", friction);
        createGroup("Numerics");
        static float timestepSize = 0.5f;
        createFloatItem("Time step size", timestepSize);
        ImGui::End();
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

void _SimulationParametersWindow::createFloatItem(std::string const& name, float& value)
{ 
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 30);
    ImGui::SliderFloat(name.c_str(), &value, 0.0f, 1.0f);
    ImGui::PopStyleVar();
    helpMarker("This is a more typical looking tree with selectable nodes.\n"
               "Click to select, CTRL+Click to toggle, click on arrows or double-click to open.");
    ImGui::Spacing();
}

void _SimulationParametersWindow::helpMarker(std::string const& text)
{
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextInfoColor);
    ImGui::Text("(?)");
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}
