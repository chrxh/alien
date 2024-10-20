#include "ResetPasswordDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "NewPasswordDialog.h"

ResetPasswordDialog::ResetPasswordDialog()
    : AlienDialog("Reset password")
{}

void ResetPasswordDialog::open(std::string const& userName, UserInfo const& userInfo)
{
    AlienDialog::open();
    _userName = userName;
    _email.clear();
    _userInfo = userInfo;
}

void ResetPasswordDialog::processIntern()
{
    AlienImGui::Text("Security information");
    AlienImGui::HelpMarker("The data transfer to the server is encrypted via https. On the server side, the email address is not stored in cleartext, but "
                           "as a SHA-256 hash value in the database.");
    AlienImGui::Text("Data privacy policy");
    AlienImGui::HelpMarker(
        "The entered e-mail address will not be passed on to third parties and is used only for the following two reasons: 1) To send the confirmation code. "
        "2) A SHA-256 hash value of the email address is stored on the server to verify that it is not yet in use.");
    AlienImGui::Separator();
    AlienImGui::Text("Please enter your email address to receive the\nconfirmation code to reset the password.");
    AlienImGui::Separator();
    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Email").textWidth(0), _email);

    AlienImGui::Separator();

    ImGui::BeginDisabled(_email.empty());
    if (AlienImGui::Button("OK")) {
        close();

        onResetPassword();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void ResetPasswordDialog::onResetPassword()
{
    if (NetworkService::get().resetPassword(_userName, _email)) {
        NewPasswordDialog::get().open(_userName, _userInfo);
    } else {
        MessageDialog::get().information(
            "Error", "An error occurred on the server. This could be related to the fact that the\nemail address is wrong.");
    }
}
