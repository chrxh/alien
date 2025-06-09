#include "ResetPasswordDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "GenericMessageDialog.h"
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
    AlienGui::Text("Security information");
    AlienGui::HelpMarker("The data transfer to the server is encrypted via https. On the server side, the email address is not stored in cleartext, but "
                           "as a SHA-256 hash value in the database.");
    AlienGui::Text("Data privacy policy");
    AlienGui::HelpMarker(
        "The entered e-mail address will not be passed on to third parties and is used only for the following two reasons: 1) To send the confirmation code. "
        "2) A SHA-256 hash value of the email address is stored on the server to verify that it is not yet in use.");
    AlienGui::Separator();
    AlienGui::Text("Please enter your email address to receive the\nconfirmation code to reset the password.");
    AlienGui::Separator();
    AlienGui::InputText(AlienGui::InputTextParameters().hint("Email").textWidth(0), _email);

    AlienGui::Separator();

    ImGui::BeginDisabled(_email.empty());
    if (AlienGui::Button("OK")) {
        close();

        onResetPassword();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void ResetPasswordDialog::onResetPassword()
{
    if (NetworkService::get().resetPassword(_userName, _email)) {
        NewPasswordDialog::get().open(_userName, _userInfo);
    } else {
        GenericMessageDialog::get().information(
            "Error", "An error occurred on the server. This could be related to the fact that the\nemail address is wrong.");
    }
}
