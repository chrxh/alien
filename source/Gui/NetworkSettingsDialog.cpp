#include "NetworkSettingsDialog.h"

#include <imgui.h>

#include "Network/NetworkController.h"

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "StyleRepository.h"

namespace
{
    auto const RightColumnWidth = 150.0f;
}

_NetworkSettingsDialog::_NetworkSettingsDialog(BrowserWindow const& browserWindow, NetworkController const& networkController)
    : _AlienDialog("Network settings")
    , _browserWindow(browserWindow)
    , _networkController(networkController)
{
}

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
    _origServerAddress = _networkController->getServerAddress();
    _serverAddress = _origServerAddress;
}

void _NetworkSettingsDialog::onChangeSettings()
{
    _networkController->setServerAddress(_serverAddress);
    _browserWindow->onRefresh();
}
