#include "LoginDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "NetworkController.h"
#include "MessageDialog.h"
#include "CreateUserDialog.h"
#include "BrowserWindow.h"
#include "ResetPasswordDialog.h"

_LoginDialog::_LoginDialog(
    BrowserWindow const& browserWindow,
    CreateUserDialog const& createUserDialog,
    ResetPasswordDialog const& resetPasswordDialog,
    NetworkController const& networkController)
    : _browserWindow(browserWindow)
    , _createUserDialog(createUserDialog)
    , _networkController(networkController)
    , _resetPasswordDialog(resetPasswordDialog)

{
    auto& settings = GlobalSettings::getInstance();
    _remember = settings.getBoolState("dialogs.login.remember", true);
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

        AlienImGui::Text("How to create a new user?");
        AlienImGui::HelpMarker("Please enter the desired user name and password and proceed by clicking the 'Create user' button.");

        AlienImGui::Text("Forgot your password?");
        AlienImGui::HelpMarker("Please enter the user name and proceed by clicking the 'Reset password' button.");

        AlienImGui::Text("Security information");
        AlienImGui::HelpMarker(
            "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
            "value in the database. If the toggle 'Remember' is activated the password will be stored in 'settings.json' on your local machine.");

        AlienImGui::Separator();

        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("User name").textWidth(0), _userName);
        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Password").password(true).textWidth(0), _password);
        AlienImGui::ToggleButton(
            AlienImGui::ToggleButtonParameters()
                .name("Remember"),
            _remember);

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
        ImGui::Dummy(ImVec2(40.0f, 0.0f));

        ImGui::SameLine();
        ImGui::BeginDisabled(_userName.empty() || _password.empty());
        if (AlienImGui::Button("Create user")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            _createUserDialog->show(_userName, _password);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(_userName.empty());
        if (AlienImGui::Button("Reset password")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            _resetPasswordDialog->show(_userName);
        }
        ImGui::EndDisabled();

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
        return;
    }
    _browserWindow->onRefresh();
}
