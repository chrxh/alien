#include "CreateUserDialog.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Network/NetworkService.h>

#include "ActivateUserDialog.h"
#include "AlienGui.h"
#include "GenericMessageDialog.h"

CreateUserDialog::CreateUserDialog()
    : AlienDialog(_("Create user"))
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
    AlienGui::Text(_("Security information"));
    AlienGui::HelpMarker(_("The data transfer to the server is encrypted via https. On the server side, the email address is not stored in cleartext, but "
                          "as a SHA-256 hash value in the database."));
    AlienGui::Text(_("Data privacy policy"));
    AlienGui::HelpMarker(_("The entered e-mail address will not be passed on to third parties and is used only for the following two reasons: 1) To send "
                          "the confirmation code. "
                          "2) A SHA-256 hash value of the email address is stored on the server to verify that it is not yet in use."));
    AlienGui::Separator();
    AlienGui::Text(_("Please enter your email address to receive the\nconfirmation code for the new user."));
    AlienGui::Separator();
    AlienGui::InputText(AlienGui::InputTextParameters().hint(_("Email")).textWidth(0), _email);

    AlienGui::Separator();

    ImGui::BeginDisabled(_email.empty());
    if (AlienGui::Button(_("Create user"))) {
        close();

        onCreateUser();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button(_("Cancel"))) {
        close();
    }
}

void CreateUserDialog::onCreateUser()
{
    if (NetworkService::get().createUser(_userName, _password, _email)) {
        ActivateUserDialog::get().open(_userName, _password, _userInfo);
    } else {
        GenericMessageDialog::get().information(
            _("Error"),
            "An error occurred on the server. This could be related to the fact that\n" ICON_FA_CARET_RIGHT
            " your user name or email address is already in use,\n" ICON_FA_CARET_RIGHT " or your user "
            "name contains white spaces.");
    }
}
