#include "CreatorWindow.h"

#include <imgui.h>

#include "IconFontCppHeaders/IconsFontAwesome5.h"

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
        if (AlienImGui::BeginToolbarButton(ICON_FA_SHAPES)) {
        }
        AlienImGui::EndToolbarButton();
        AlienImGui::Group("General properties");
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
