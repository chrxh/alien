#include "LoginDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "NetworkController.h"
#include "MessageDialog.h"

_LoginDialog::_LoginDialog(NetworkController const& networkController)
    : _networkController(networkController)
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

        AlienImGui::InputText(AlienImGui::InputTextParameters().name("User name"), _userName);
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Password").password(true), _password);
        AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Remember").tooltip("Only hash values of the password will be saved."), _remember);

        AlienImGui::Separator();

        ImGui::BeginDisabled(_userName.empty() || _password.empty());
        if (AlienImGui::Button("OK")) {
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
