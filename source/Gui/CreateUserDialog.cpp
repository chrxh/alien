#include "CreateUserDialog.h"

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "ActivateUserDialog.h"

CreateUserDialog::CreateUserDialog()
    : AlienDialog("Create user")
{}

void CreateUserDialog::open(std::string const& userName, std::string const& password, UserInfo const& userInfo)
{
    AlienDialog::open();
    _userName = userName;
    _password = password;
    _email.clear();
    _userInfo = userInfo;
}

void CreateUserDialog::processIntern()
{
    AlienImGui::Text("Security information");
    AlienImGui::HelpMarker("The data transfer to the server is encrypted via https. On the server side, the email address is not stored in cleartext, but "
                           "as a SHA-256 hash value in the database.");
    AlienImGui::Text("Data privacy policy");
    AlienImGui::HelpMarker("The entered e-mail address will not be passed on to third parties and is used only for the following two reasons: 1) To send "
                           "the confirmation code. "
                           "2) A SHA-256 hash value of the email address is stored on the server to verify that it is not yet in use.");
    AlienImGui::Separator();
    AlienImGui::Text("Please enter your email address to receive the\nconfirmation code for the new user.");
    AlienImGui::Separator();
    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Email").textWidth(0), _email);

    AlienImGui::Separator();

    ImGui::BeginDisabled(_email.empty());
    if (AlienImGui::Button("Create user")) {
        close();

        onCreateUser();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void CreateUserDialog::onCreateUser()
{
    if (NetworkService::get().createUser(_userName, _password, _email)) {
        ActivateUserDialog::get().open(_userName, _password, _userInfo);
    } else {
        MessageDialog::get().information(
            "Error",
            "An error occurred on the server. This could be related to the fact that\n" ICON_FA_CARET_RIGHT
            " your user name or email address is already in use,\n" ICON_FA_CARET_RIGHT " or your user "
            "name contains white spaces.");
    }
}
