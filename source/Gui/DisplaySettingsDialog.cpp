#include "DisplaySettingsDialog.h"

#include <sstream>

#include <GLFW/glfw3.h>
#include "imgui.h"

#include "Base/LoggingService.h"
#include "Base/ServiceLocator.h"

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "WindowController.h"

namespace
{
    auto const ItemTextWidth = 130.0f;
}

_DisplaySettingsDialog::_DisplaySettingsDialog(WindowController const& windowController)
    : _windowController(windowController)
{
}

_DisplaySettingsDialog::~_DisplaySettingsDialog()
{
}

void _DisplaySettingsDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Display settings");
    if (ImGui::BeginPopupModal("Display settings", NULL, ImGuiWindowFlags_None)) {

        auto isFullscreen = _windowController->isFullscreen();
        if (ImGui::Checkbox("Full screen", &isFullscreen)) {
            _windowController->setFullscreen(isFullscreen);
        }

/*
        ImGui::BeginDisabled(!_fullscreen);

        auto prevVideoModeSelection = _videoModeSelection;
        AlienImGui::Combo(
            AlienImGui::ComboParameters()
                .name("Resolution")
                .textWidth(ItemTextWidth)
                .defaultValue(_origVideoModeSelection)
                .values(createVideoModeStrings()),
            _videoModeSelection);

        if (prevVideoModeSelection != _videoModeSelection) {
            onSetVideoMode();
        }
        ImGui::EndDisabled();
*/

        AlienImGui::Separator();

        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            _windowController->setFullscreen(_origFullscreen);
        }

        ImGui::EndPopup();
    }
}

void _DisplaySettingsDialog::show()
{
    _show = true;
    _origFullscreen = _windowController->isFullscreen();
}
