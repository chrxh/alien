#include "CreateUserDialog.h"

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Network/NetworkController.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "ActivateUserDialog.h"

_CreateUserDialog::_CreateUserDialog(ActivateUserDialog const& activateUserDialog, NetworkController const& networkController)
    : _AlienDialog("Create user")
    , _networkController(networkController)
    , _activateUserDialog(activateUserDialog)
{
}

_CreateUserDialog::~_CreateUserDialog()
{
}

void _CreateUserDialog::open(std::string const& userName, std::string const& password, UserInfo const& userInfo)
{
    _AlienDialog::open();
    _userName = userName;
    _password = password;
    _email.clear();
    _userInfo = userInfo;
}

void _CreateUserDialog::processIntern()
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

void _CreateUserDialog::onCreateUser()
{
    if (_networkController->createUser(_userName, _password, _email)) {
        _activateUserDialog->open(_userName, _password, _userInfo);
    } else {
        MessageDialog::getInstance().information(
            "Error",
            "An error occurred on the server. This could be related to the fact that\n" ICON_FA_CARET_RIGHT
            " your user name or email address is already in use,\n" ICON_FA_CARET_RIGHT " or your user "
            "name contains white spaces.");
    }
}
