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
#include "LoginController.h"

_LoginDialog::_LoginDialog(
    SimulationController const& simController,
    PersisterController const& persisterController,
    BrowserWindow const& browserWindow,
    CreateUserDialog const& createUserDialog,
    ActivateUserDialog const& activateUserDialog,
    ResetPasswordDialog const& resetPasswordDialog)
    : _AlienDialog("Login")
    , _simController(simController)
    , _persisterController(persisterController)
    , _browserWindow(browserWindow)
    , _createUserDialog(createUserDialog)
    , _activateUserDialog(activateUserDialog)
    , _resetPasswordDialog(resetPasswordDialog)

{
}

_LoginDialog::~_LoginDialog()
{
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

    auto remember = LoginController::get().isRemember();
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Remember").tooltip(Const::LoginRememberTooltip), remember);
    LoginController::get().setRemember(remember);

    auto shareGpuInfo = LoginController::get().shareGpuInfo();
    AlienImGui::ToggleButton(
        AlienImGui::ToggleButtonParameters()
            .name("Share GPU model info")
            .tooltip(Const::LoginShareGpuInfoTooltip1 + _simController->getGpuName() + "\n" + Const::LoginShareGpuInfoTooltip2),
        shareGpuInfo);
    LoginController::get().setShareGpuInfo(shareGpuInfo);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    ImGui::BeginDisabled(_userName.empty() || _password.empty());
    if (AlienImGui::Button("Login")) {
        close();
        onLogin();
        if (!remember) {
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
        _createUserDialog->open(_userName, _password, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(_userName.empty());
    if (AlienImGui::Button("Reset password")) {
        close();
        _resetPasswordDialog->open(_userName, LoginController::get().getUserInfo());
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

    auto userInfo = LoginController::get().getUserInfo();

    if (!NetworkService::login(errorCode, _userName, _password, userInfo)) {
        switch (errorCode) {
        case LoginErrorCode_UnknownUser: {
            _activateUserDialog->open(_userName, _password, userInfo);
        } break;
        default: {
            MessageDialog::getInstance().information("Error", "Login failed.");
        } break;
        }
        return;
    }
    _browserWindow->onRefresh();
    LoginController::get().saveSettings();
}
