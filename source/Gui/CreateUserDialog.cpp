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

        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("User name").textWidth(0), _userName);
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().hint("Email").password(true).textWidth(0).tooltip(
                "The email address will not be stored on the server and only used to send the confirmation code."),
            _password);
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Password").password(true).textWidth(0), _password);
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Re-enter password").password(true).textWidth(0), _password);
        
        AlienImGui::Separator();
        AlienImGui::Text("Security information");
        AlienImGui::HelpMarker(
            "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
            "value in the database. The email address will not be stored and only used to send the confirmation code.");
        AlienImGui::Separator();

        ImGui::BeginDisabled(_userName.empty() || _password.empty());
        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
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

void _CreateUserDialog::show()
{
    _show = true;
}
