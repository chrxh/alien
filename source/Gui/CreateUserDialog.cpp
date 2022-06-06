#include "CreateUserDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "NetworkController.h"

_CreateUserDialog::_CreateUserDialog(NetworkController const& networkController)
    : _networkController(networkController)
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
        AlienImGui::Text("Please enter your email address to receive the\nactivation code for the new user.");
        AlienImGui::Separator();
        AlienImGui::Text("Security information");
        AlienImGui::HelpMarker("The data transfer to the server is encrypted via https. On the server side, the email address is not stored in cleartext, but "
                               "as a SHA-256 hash value in the database.");
        AlienImGui::Text("Data privacy policy");
        AlienImGui::HelpMarker("The email address will only be used to send the activation code. It will not be passed on to third parties. A SHA-256 "
                               "hash value of the email address is stored on the server side to check uniqueness.");
        AlienImGui::Separator();
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().hint("Email").textWidth(0),
            _email);
        
        AlienImGui::Separator();

        ImGui::BeginDisabled(_email.empty());
        if (AlienImGui::Button("OK")) {
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
}

void _CreateUserDialog::onCreateUser()
{
    _networkController->createUser(_userName, _password, _email);
}
