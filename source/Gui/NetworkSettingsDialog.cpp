#include "NetworkSettingsDialog.h"

#include <imgui.h>

#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "StyleRepository.h"

namespace
{
    auto const RightColumnWidth = 150.0f;
}

NetworkSettingsDialog::NetworkSettingsDialog()
    : AlienDialog(_("Network settings"))
{}

void NetworkSettingsDialog::processIntern()
{
    AlienGui::InputText(
        AlienGui::InputTextParameters().name(_("Blocks")).defaultValue(_origServerAddress).name(_("Server address")).textWidth(RightColumnWidth), _serverAddress);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button(_("Adopt"))) {
        close();
        onChangeSettings();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button(_("Cancel"))) {
        close();
    }
}

void NetworkSettingsDialog::openIntern()
{
    _origServerAddress = NetworkService::get().getServerAddress();
    _serverAddress = _origServerAddress;
}

void NetworkSettingsDialog::onChangeSettings()
{
    NetworkService::get().setServerAddress(_serverAddress);
    BrowserWindow::get().onRefresh();
}
