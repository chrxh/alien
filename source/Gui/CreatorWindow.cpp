#include "CreatorWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "AlienImGui.h"

_CreatorWindow::_CreatorWindow()
{
    _on = GlobalSettings::getInstance().getBoolState("windows.creator.active", true);
}

_CreatorWindow::~_CreatorWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.creator.active", _on);
}

void _CreatorWindow::process()
{
    if (!_on) {
        return;
    }
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Creator", &_on)) {
        if (AlienImGui::BeginToolbarButton(ICON_FA_SUN)) {
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_FA_ATOM)) {
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_RECTANGLE)) {
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_HEXAGON)) {
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_DISC)) {
        }
        AlienImGui::EndToolbarButton();

        AlienImGui::Group("General properties");
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").format("%.2f"), _energy);
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Cell distance").format("%.2f").step(0.1), _distance);
    }
    ImGui::End();
}

bool _CreatorWindow::isOn() const
{
    return _on;
}

void _CreatorWindow::setOn(bool value)
{
    _on = value;
}
