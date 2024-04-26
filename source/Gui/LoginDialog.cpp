#include "LoginDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SimulationController.h"
#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "CreateUserDialog.h"
#include "BrowserWindow.h"
#include "ResetPasswordDialog.h"
#include "ActivateUserDialog.h"
#include "StyleRepository.h"
#include "HelpStrings.h"

_LoginDialog::_LoginDialog(
    SimulationController const& simController,
    BrowserWindow const& browserWindow,
    CreateUserDialog const& createUserDialog,
    ActivateUserDialog const& activateUserDialog,
    ResetPasswordDialog const& resetPasswordDialog)
    : _AlienDialog("Login")
    , _simController(simController)
    , _browserWindow(browserWindow)
    , _createUserDialog(createUserDialog)
    , _activateUserDialog(activateUserDialog)
    , _resetPasswordDialog(resetPasswordDialog)

{
    auto& settings = GlobalSettings::getInstance();
    _remember = settings.getBool("dialogs.login.remember", _remember);
    _shareGpuInfo = settings.getBool("dialogs.login.share gpu info", _shareGpuInfo);

    if (_remember) {
        _userName = settings.getString("dialogs.login.user name", "");
        _password = settings.getString("dialogs.login.password", "");
        if (!_userName.empty()) {
            LoginErrorCode errorCode;
            if (!NetworkService::login(errorCode, _userName, _password, getUserInfo())) {
                if (errorCode != LoginErrorCode_UnconfirmedUser) {
                    MessageDialog::getInstance().information("Error", "Login failed.");
                }
            }
        }
    }
}

_LoginDialog::~_LoginDialog()
{
    saveSettings();
}

bool _LoginDialog::isShareGpuInfo() const
{
    return _shareGpuInfo;
}

void _LoginDialog::processIntern()
{
    AlienImGui::Text("How to create a new user?");
    AlienImGui::HelpMarker(Const::LoginHowToCreateNewUseTooltip);

    AlienImGui::Text("Forgot your password?");
    AlienImGui::HelpMarker(Const::LoginForgotYourPasswordTooltip);

    AlienImGui::Text("Security information");
    AlienImGui::HelpMarker(Const::LoginSecurityInformationTooltip);

    AlienImGui::Separator();

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("User name").textWidth(0), _userName);
    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Password").password(true).textWidth(0), _password);
    AlienImGui::Separator();
    ImGui::Spacing();
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Remember").tooltip(Const::LoginRememberTooltip), _remember);
    AlienImGui::ToggleButton(
        AlienImGui::ToggleButtonParameters()
            .name("Share GPU model info")
            .tooltip(Const::LoginShareGpuInfoTooltip1 + _simController->getGpuName() + "\n" + Const::LoginShareGpuInfoTooltip2),
        _shareGpuInfo);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
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
        _createUserDialog->open(_userName, _password, getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(_userName.empty());
    if (AlienImGui::Button("Reset password")) {
        close();
        _resetPasswordDialog->open(_userName, getUserInfo());
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

    auto userInfo = getUserInfo();

    if (!NetworkService::login(errorCode, _userName, _password, userInfo)) {
        switch (errorCode) {
        case LoginErrorCode_UnconfirmedUser: {
            _activateUserDialog->open(_userName, _password, userInfo);
        } break;
        default: {
            MessageDialog::getInstance().information("Error", "Login failed.");
        } break;
        }
        return;
    }
    _browserWindow->onRefresh();
    saveSettings();
}

void _LoginDialog::saveSettings()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setBool("dialogs.login.remember", _remember);
    settings.setBool("dialogs.login.share gpu info", _shareGpuInfo);
    if (_remember) {
        settings.setString("dialogs.login.user name", _userName);
        settings.setString("dialogs.login.password", _password);
    }
}

UserInfo _LoginDialog::getUserInfo()
{
    UserInfo result;
    if (_shareGpuInfo) {
        result.gpu = _simController->getGpuName();
    }
    return result;
}
