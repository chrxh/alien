#include "NetworkSettingsDialog.h"

#include <imgui.h>

#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "StyleRepository.h"

namespace
{
    auto const RightColumnWidth = 150.0f;
}

_NetworkSettingsDialog::_NetworkSettingsDialog()
    : AlienDialog("Network settings")
{}

void _NetworkSettingsDialog::processIntern()
{
    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name("Blocks").defaultValue(_origServerAddress).name("Server address").textWidth(RightColumnWidth), _serverAddress);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        close();
        onChangeSettings();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void _NetworkSettingsDialog::openIntern()
{
    _origServerAddress = NetworkService::get().getServerAddress();
    _serverAddress = _origServerAddress;
}

void _NetworkSettingsDialog::onChangeSettings()
{
    NetworkService::get().setServerAddress(_serverAddress);
    BrowserWindow::get().onRefresh();
}
