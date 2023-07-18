#include "LoginDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "NetworkController.h"
#include "MessageDialog.h"
#include "CreateUserDialog.h"
#include "BrowserWindow.h"
#include "ResetPasswordDialog.h"
#include "ActivateUserDialog.h"
#include "StyleRepository.h"

_LoginDialog::_LoginDialog(
    BrowserWindow const& browserWindow,
    CreateUserDialog const& createUserDialog,
    ActivateUserDialog const& activateUserDialog,
    ResetPasswordDialog const& resetPasswordDialog,
    NetworkController const& networkController)
    : _AlienDialog("Login")
    , _browserWindow(browserWindow)
    , _createUserDialog(createUserDialog)
    , _activateUserDialog(activateUserDialog)
    , _networkController(networkController)
    , _resetPasswordDialog(resetPasswordDialog)

{
    auto& settings = GlobalSettings::getInstance();
    _remember = settings.getBoolState("dialogs.login.remember", true);
    if (_remember) {
        _userName = settings.getStringState("dialogs.login.user name", "");
        _password = settings.getStringState("dialogs.login.password", "");
        if (!_userName.empty()) {
            LoginErrorCode errorCode;
            if (!_networkController->login(errorCode, _userName, _password)) {
                if (errorCode != LoginErrorCode_UnconfirmedUser) {
                    MessageDialog::getInstance().show("Error", "Login failed.");
                }
            }
        }
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

void _LoginDialog::processIntern()
{
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
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Remember"), _remember);

    AlienImGui::Separator();

    ImGui::BeginDisabled(_userName.empty() || _password.empty());
    if (AlienImGui::Button("Login")) {
        close();
        onLogin();
        if (!_remember) {
            _userName.clear();
            _password.clear();
        }
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    AlienImGui::VerticalSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(_userName.empty() || _password.empty());
    if (AlienImGui::Button("Create user")) {
        close();
        _createUserDialog->open(_userName, _password);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(_userName.empty());
    if (AlienImGui::Button("Reset password")) {
        close();
        _resetPasswordDialog->open(_userName);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienImGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void _LoginDialog::onLogin()
{
    LoginErrorCode errorCode;
    if (!_networkController->login(errorCode, _userName, _password)) {
        switch (errorCode) {
        case LoginErrorCode_UnconfirmedUser: {
            _activateUserDialog->open(_userName, _password);
        } break;
        default: {
            MessageDialog::getInstance().show("Error", "Login failed.");
        } break;
        }
        return;
    }
    _browserWindow->onRefresh();
}
