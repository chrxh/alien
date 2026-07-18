#include "ActivateUserDialog.h"

#include <imgui.h>

#include <Network/NetworkService.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"

void ActivateUserDialog::initIntern()
{

}

void ActivateUserDialog::open(std::string const& userName, std::string const& password, UserInfo const& userInfo)
{
    AlienDialog::open();
    _userName = userName;
    _password = password;
    _userInfo = userInfo;
}

ActivateUserDialog::ActivateUserDialog()
    : AlienDialog(_("Activate user"))
{}

void ActivateUserDialog::processIntern()
{
    AlienGui::Text(_("Please enter the confirmation code sent to your email address."));
    AlienGui::HelpMarker(
        _("Please check your spam folder if you did not find an email. If you did not receive an email there, try signing up with possibly another "
          "email address. If this still does not work, please contact info@alien-project.org."));
    AlienGui::Separator();
    AlienGui::InputText(AlienGui::InputTextParameters().hint(_("Code (case sensitive)")).textWidth(0), _confirmationCode);

    AlienGui::Separator();

    ImGui::BeginDisabled(_confirmationCode.empty());
    if (AlienGui::Button(_("OK"))) {
        close();
        onActivateUser();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    AlienGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienGui::Button(_("Resend"))) {
        CreateUserDialog::get().onCreateUser();
    }

    ImGui::SameLine();
    if (AlienGui::Button(_("Resend to other email address"))) {
        close();
        CreateUserDialog::get().open(_userName, _password, _userInfo);
    }

    ImGui::SameLine();
    AlienGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienGui::Button(_("Cancel"))) {
        close();
    }
}

void ActivateUserDialog::onActivateUser()
{
    auto result = NetworkService::get().activateUser(_userName, _password, _userInfo, _confirmationCode);
    if (result) {
        LoginErrorCode errorCode;
        result |= NetworkService::get().login(errorCode, _userName, _password, _userInfo);
    }
    if (!result) {
        GenericMessageDialog::get().information(_("Error"), _("An error occurred on the server. Your entered code may be incorrect.\nPlease try to register again."));
    } else {
        GenericMessageDialog::get().information(
            _("Information"),
            "The user '" + _userName
                + "' has been successfully created.\nYou are logged in and are now able to upload your own simulations\nor upvote others by likes.");
        BrowserWindow::get().onRefresh();
    }
}
