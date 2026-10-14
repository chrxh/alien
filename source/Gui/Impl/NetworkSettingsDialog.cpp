#include "NetworkSettingsDialog.h"

#include <imgui.h>

#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "HelpStrings.h"
#include "StyleService.h"

namespace
{
    auto const RightColumnWidth = 150.0f;
}

NetworkSettingsDialog::NetworkSettingsDialog()
    : AlienDialog("Network settings")
{}

void NetworkSettingsDialog::processIntern()
{
    AlienGui::InputText(
        AlienGui::InputTextParameters().name("Blocks").defaultValue(_origServerAddress).name("Server address").textWidth(RightColumnWidth), _serverAddress);
    AlienGui::Checkbox(
        AlienGui::CheckboxParameters().name("Count me as active").textWidth(RightColumnWidth).tooltip(Const::NetworkCountAsActiveTooltip), &_countAsActive);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        close();
        onChangeSettings();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void NetworkSettingsDialog::openIntern()
{
    _origServerAddress = NetworkService::get().getServerAddress();
    _serverAddress = _origServerAddress;
    _countAsActive = NetworkService::get().isCountAsActive();
}

void NetworkSettingsDialog::onChangeSettings()
{
    NetworkService::get().setServerAddress(_serverAddress);
    NetworkService::get().setCountAsActive(_countAsActive);
    BrowserWindow::get().onRefresh();
}
