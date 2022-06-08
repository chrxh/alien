#include "NewPasswordDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "NetworkController.h"

_NewPasswordDialog::_NewPasswordDialog(BrowserWindow const& browserWindow, NetworkController const& networkController)
    : _browserWindow(browserWindow)
    , _networkController(networkController)
{}

_NewPasswordDialog::~_NewPasswordDialog() {}

void _NewPasswordDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("New password");
    if (ImGui::BeginPopupModal("New password", NULL, ImGuiWindowFlags_None)) {
        AlienImGui::Text("Security information");
        AlienImGui::HelpMarker(
            "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
            "value in the database.");

        AlienImGui::Separator();

        AlienImGui::Text("Please enter a new password and the confirmation code\nsent to your email address.");
        AlienImGui::Separator();
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("New password").password(true).textWidth(0), _newPassword);
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Code (case sensitive)").textWidth(0), _confirmationCode);

        AlienImGui::Separator();

        ImGui::BeginDisabled(_confirmationCode.empty());
        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            onNewPassword();
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

void _NewPasswordDialog::show(std::string const& userName)
{
    _show = true;
    _userName = userName;
    _newPassword.clear();
    _confirmationCode.clear();
}

void _NewPasswordDialog::onNewPassword()
{
    auto result = _networkController->setNewPassword(_userName, _newPassword, _confirmationCode);
    if (result) {
        result |= _networkController->login(_userName, _newPassword);
    }
    if (!result) {
        MessageDialog::getInstance().show("Error", "An error occurred on the server. Your entered code may be incorrect.\nPlease try to reset the password again.");
        return;
    }
    MessageDialog::getInstance().show("Information", "The password has been successfully set.\nYou are logged in.");
    _browserWindow->onRefresh();
}
