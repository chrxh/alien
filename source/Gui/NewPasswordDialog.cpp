#include "NewPasswordDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationFacade.h"
#include "Network/NetworkService.h"

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "GenericMessageDialog.h"

void NewPasswordDialog::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void NewPasswordDialog::open(std::string const& userName, UserInfo const& userInfo)
{
    AlienDialog::open();
    _userName = userName;
    _newPassword.clear();
    _confirmationCode.clear();
    _userInfo = userInfo;
}

NewPasswordDialog::NewPasswordDialog()
    : AlienDialog("New password")
{}

void NewPasswordDialog::processIntern()
{
    AlienGui::Text("Security information");
    AlienGui::HelpMarker(
        "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
        "value in the database.");

    AlienGui::Separator();

    AlienGui::Text("Please enter a new password and the confirmation code\nsent to your email address.");
    AlienGui::Separator();
    AlienGui::InputText(AlienGui::InputTextParameters().hint("New password").password(true).textWidth(0), _newPassword);
    AlienGui::InputText(AlienGui::InputTextParameters().hint("Code (case sensitive)").textWidth(0), _confirmationCode);

    AlienGui::Separator();

    ImGui::BeginDisabled(_confirmationCode.empty());
    if (AlienGui::Button("OK")) {
        close();
        onNewPassword();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void NewPasswordDialog::onNewPassword()
{
    auto result = NetworkService::get().setNewPassword(_userName, _newPassword, _confirmationCode);
    if (result) {
        LoginErrorCode errorCode;
        result |= NetworkService::get().login(errorCode, _userName, _newPassword, _userInfo);
    }
    if (!result) {
        GenericMessageDialog::get().information("Error", "An error occurred on the server. Your entered code may be incorrect.\nPlease try to reset the password again.");
        return;
    }
    GenericMessageDialog::get().information("Information", "The password has been successfully set.\nYou are logged in.");
    BrowserWindow::get().onRefresh();
}
