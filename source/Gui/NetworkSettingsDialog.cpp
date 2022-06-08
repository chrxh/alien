#include "NetworkSettingsDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "NetworkController.h"
#include "BrowserWindow.h"

namespace
{
    auto const MaxContentTextWidth = 150.0f;
}

_NetworkSettingsDialog::_NetworkSettingsDialog(BrowserWindow const& browserWindow, NetworkController const& networkController)
    : _browserWindow(browserWindow), _networkController(networkController)
{
}

_NetworkSettingsDialog::~_NetworkSettingsDialog()
{
}

void _NetworkSettingsDialog::process()
{
    if (!_show) {
        return;
    }
    ImGui::OpenPopup("Network settings");
    if (ImGui::BeginPopupModal("Network settings", NULL, ImGuiWindowFlags_None)) {
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Blocks").defaultValue(_origServerAddress).name("Server address").textWidth(MaxContentTextWidth),
            _serverAddress);

        AlienImGui::Separator();

        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            onChangeSettings();
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }

        ImGui::EndPopup();
    }
}

void _NetworkSettingsDialog::show()
{
    _show = true;
    _origServerAddress = _networkController->getServerAddress();
    _serverAddress = _origServerAddress;
}

void _NetworkSettingsDialog::onChangeSettings()
{
    _networkController->setServerAddress(_serverAddress);
    _browserWindow->onRefresh();
}
