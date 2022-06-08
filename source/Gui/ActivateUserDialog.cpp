#include "ActivateUserDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "NetworkController.h"
#include "BrowserWindow.h"

_ActivateUserDialog::_ActivateUserDialog(BrowserWindow const& browserWindow, NetworkController const& networkController)
    : _browserWindow(browserWindow)
    , _networkController(networkController)
{}

_ActivateUserDialog::~_ActivateUserDialog() {}

void _ActivateUserDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Activate user");
    if (ImGui::BeginPopupModal("Activate user", NULL, ImGuiWindowFlags_None)) {
        AlienImGui::Text("Please enter the confirmation code sent to your email address.");
        AlienImGui::Separator();
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Code (case sensitive)").textWidth(0), _confirmationCode);

        AlienImGui::Separator();

        ImGui::BeginDisabled(_confirmationCode.empty());
        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            onActivateUser();
        }
        ImGui::EndDisabled();
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }

        ImGui::EndPopup();
    }
}

void _ActivateUserDialog::show(std::string const& userName, std::string const& password)
{
    _show = true;
    _userName = userName;
    _password = password;
}

void _ActivateUserDialog::onActivateUser()
{
    auto result = _networkController->activateUser(_userName, _password, _confirmationCode);
    if (result) {
        result |= _networkController->login(_userName, _password);
    }
    if (!result) {
        MessageDialog::getInstance().show("Error", "An error occurred on the server. Your entered code may be incorrect.\nPlease try to register again.");
        return;
    }
    MessageDialog::getInstance().show("Information", "The user '" + _userName + "' has been successfully created.\nYou are logged in.");
    _browserWindow->onRefresh();
}
