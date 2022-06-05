#include "LoginDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "NetworkController.h"
#include "MessageDialog.h"
#include "CreateUserDialog.h"

_LoginDialog::_LoginDialog(CreateUserDialog const& createUserDialog, NetworkController const& networkController)
    : _createUserDialog(createUserDialog)
    , _networkController(networkController)
{
    auto& settings = GlobalSettings::getInstance();
    _remember = settings.getBoolState("dialogs.login.remember", false);
    if (_remember) {
        settings.getStringState("dialogs.login.user name", "");
        settings.getStringState("dialogs.login.password", "");
    }
}

_LoginDialog::~_LoginDialog()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setBoolState("dialogs.login.remember", _remember);
    if (_remember) {
        settings.setStringState("dialogs.login.user name", _userName);
        settings.setStringState("dialogs.login.password", _password);
    }
}

void _LoginDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Login");
    if (ImGui::BeginPopupModal("Login", NULL, ImGuiWindowFlags_None)) {

        AlienImGui::Text("No user registered?");
        ImGui::SameLine();
        if (AlienImGui::Button("Create user")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            _createUserDialog->show();
        }
        AlienImGui::Separator();

        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("User name").textWidth(0), _userName);
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Password").password(true).textWidth(0), _password);
        AlienImGui::ToggleButton(
            AlienImGui::ToggleButtonParameters()
                .name("Remember")
                .tooltip("If activated the login data will be stored in 'settings.json'."),
            _remember);

        AlienImGui::Separator();
        AlienImGui::Text("Security information");
        AlienImGui::HelpMarker(
            "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
            "value in the database.");
        AlienImGui::Separator();

        ImGui::BeginDisabled(_userName.empty() || _password.empty());
        if (AlienImGui::Button("Login")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            onLogin();
            if(!_remember) {
                _userName.clear();
                _password.clear();
            }
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

void _LoginDialog::show()
{
    _show = true;
}

void _LoginDialog::onLogin()
{
    if (!_networkController->login(_userName, _password)) {
        MessageDialog::getInstance().show("Error", "Login failed.");
    }
}
