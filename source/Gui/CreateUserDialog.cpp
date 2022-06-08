#include "CreateUserDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "NetworkController.h"
#include "ActivateUserDialog.h"

_CreateUserDialog::_CreateUserDialog(ActivateUserDialog const& activateUserDialog, NetworkController const& networkController)
    : _networkController(networkController)
    , _activateUserDialog(activateUserDialog)
{
}

_CreateUserDialog::~_CreateUserDialog()
{
}

void _CreateUserDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Create user");
    if (ImGui::BeginPopupModal("Create user", NULL, ImGuiWindowFlags_None)) {
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
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().hint("Email").textWidth(0),
            _email);
        
        AlienImGui::Separator();

        ImGui::BeginDisabled(_email.empty());
        if (AlienImGui::Button("Create user")) {
            ImGui::CloseCurrentPopup();
            _show = false;

            onCreateUser();
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

void _CreateUserDialog::show(std::string const& userName, std::string const& password)
{
    _show = true;
    _userName = userName;
    _password = password;
    _email.clear();
}

void _CreateUserDialog::onCreateUser()
{
    if(_networkController->createUser(_userName, _password, _email)) {
        _activateUserDialog->show(_userName, _password);
    } else {
        MessageDialog::getInstance().show(
            "Error", "An error occurred on the server. This could be related to the fact that the\nuser name or email address is already in use.");
    }
}
