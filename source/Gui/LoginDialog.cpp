#include "LoginDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SimulationFacade.h"
#include "Network/NetworkService.h"

#include "AlienGui.h"
#include "GenericMessageDialog.h"
#include "CreateUserDialog.h"
#include "BrowserWindow.h"
#include "ResetPasswordDialog.h"
#include "ActivateUserDialog.h"
#include "StyleRepository.h"
#include "HelpStrings.h"
#include "LoginController.h"

void LoginDialog::initIntern(SimulationFacade simulationFacade, PersisterFacade persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade = persisterFacade;
}

LoginDialog::LoginDialog()
    : AlienDialog("Login")
{}

void LoginDialog::processIntern()
{
    AlienGui::Text("How to create a new user?");
    AlienGui::HelpMarker(Const::LoginHowToCreateNewUseTooltip);

    AlienGui::Text("Forgot your password?");
    AlienGui::HelpMarker(Const::LoginForgotYourPasswordTooltip);

    AlienGui::Text("Security information");
    AlienGui::HelpMarker(Const::LoginSecurityInformationTooltip);

    AlienGui::Separator();

    auto& loginController = LoginController::get();
    auto userName = loginController.getUserName();
    AlienGui::InputText(AlienGui::InputTextParameters().hint("User name").textWidth(0), userName);
    loginController.setUserName(userName);

    auto password= loginController.getPassword();
    AlienGui::InputText(AlienGui::InputTextParameters().hint("Password").password(true).textWidth(0), password);
    loginController.setPassword(password);

    AlienGui::Separator();
    ImGui::Spacing();

    auto remember = loginController.isRemember();
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Remember").tooltip(Const::LoginRememberTooltip), remember);
    loginController.setRemember(remember);

    auto shareGpuInfo = loginController.shareGpuInfo();
    AlienGui::ToggleButton(
        AlienGui::ToggleButtonParameters()
            .name("Share GPU model info")
            .tooltip(Const::LoginShareGpuInfoTooltip1 + _simulationFacade->getGpuName() + "\n" + Const::LoginShareGpuInfoTooltip2),
        shareGpuInfo);
    loginController.setShareGpuInfo(shareGpuInfo);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    ImGui::BeginDisabled(userName.empty() || password.empty());
    if (AlienGui::Button("Login")) {
        close();
        loginController.onLogin();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    AlienGui::VerticalSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(userName.empty() || password.empty());
    if (AlienGui::Button("Create user")) {
        close();
        CreateUserDialog::get().open(userName, password, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(userName.empty());
    if (AlienGui::Button("Reset password")) {
        close();
        ResetPasswordDialog::get().open(userName, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}
