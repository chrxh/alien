#include "LoginDialog.h"

#include <imgui.h>

#include <Base/GlobalSettings.h>

#include <Network/NetworkService.h>

#include <EngineInterface/SimulationFacade.h>

#include "ActivateUserDialog.h"
#include "AlienGui.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "GenericMessageDialog.h"
#include "HelpStrings.h"
#include "LoginController.h"
#include "ResetPasswordDialog.h"
#include "StyleRepository.h"
#include <EngineInterface/SimulationFacade.h>

void LoginDialog::initIntern()
{

}

LoginDialog::LoginDialog()
    : AlienDialog(_("Login"))
{}

void LoginDialog::processIntern()
{
    AlienGui::Text(_("How to create a new user?"));
    AlienGui::HelpMarker(Const::LoginHowToCreateNewUseTooltip);

    AlienGui::Text(_("Forgot your password?"));
    AlienGui::HelpMarker(Const::LoginForgotYourPasswordTooltip);

    AlienGui::Text(_("Security information"));
    AlienGui::HelpMarker(Const::LoginSecurityInformationTooltip);

    AlienGui::Separator();

    auto& loginController = LoginController::get();
    auto userName = loginController.getUserName();
    AlienGui::InputText(AlienGui::InputTextParameters().hint(_("User name")).textWidth(0), userName);
    loginController.setUserName(userName);

    auto password = loginController.getPassword();
    AlienGui::InputText(AlienGui::InputTextParameters().hint(_("Password")).password(true).textWidth(0), password);
    loginController.setPassword(password);

    AlienGui::Separator();
    ImGui::Spacing();

    auto remember = loginController.isRemember();
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name(_("Remember")).tooltip(Const::LoginRememberTooltip), remember);
    loginController.setRemember(remember);

    auto shareGpuInfo = loginController.shareGpuInfo();
    AlienGui::ToggleButton(
        AlienGui::ToggleButtonParameters()
            .name(_("Share GPU model info"))
            .tooltip(Const::LoginShareGpuInfoTooltip1 + _SimulationFacade::get()->getGpuName() + "\n" + Const::LoginShareGpuInfoTooltip2),
        shareGpuInfo);
    loginController.setShareGpuInfo(shareGpuInfo);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    ImGui::BeginDisabled(userName.empty() || password.empty());
    if (AlienGui::Button(_("Login"))) {
        close();
        loginController.onLogin();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    AlienGui::VerticalSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(userName.empty() || password.empty());
    if (AlienGui::Button(_("Create user"))) {
        close();
        CreateUserDialog::get().open(userName, password, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(userName.empty());
    if (AlienGui::Button(_("Reset password"))) {
        close();
        ResetPasswordDialog::get().open(userName, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienGui::Button(_("Cancel"))) {
        close();
    }
}
