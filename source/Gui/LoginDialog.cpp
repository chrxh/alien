#include "LoginDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SimulationFacade.h"
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
    SimulationFacade const& simulationFacade,
    PersisterFacade const& persisterFacade,
    CreateUserDialog const& createUserDialog,
    ActivateUserDialog const& activateUserDialog,
    ResetPasswordDialog const& resetPasswordDialog)
    : _AlienDialog("Login")
    , _simulationFacade(simulationFacade)
    , _persisterFacade(persisterFacade)
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

    auto& loginController = LoginController::get();
    auto userName = loginController.getUserName();
    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("User name").textWidth(0), userName);
    loginController.setUserName(userName);

    auto password= loginController.getPassword();
    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Password").password(true).textWidth(0), password);
    loginController.setPassword(password);

    AlienImGui::Separator();
    ImGui::Spacing();

    auto remember = loginController.isRemember();
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Remember").tooltip(Const::LoginRememberTooltip), remember);
    loginController.setRemember(remember);

    auto shareGpuInfo = loginController.shareGpuInfo();
    AlienImGui::ToggleButton(
        AlienImGui::ToggleButtonParameters()
            .name("Share GPU model info")
            .tooltip(Const::LoginShareGpuInfoTooltip1 + _simulationFacade->getGpuName() + "\n" + Const::LoginShareGpuInfoTooltip2),
        shareGpuInfo);
    loginController.setShareGpuInfo(shareGpuInfo);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    ImGui::BeginDisabled(userName.empty() || password.empty());
    if (AlienImGui::Button("Login")) {
        close();
        loginController.onLogin();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    AlienImGui::VerticalSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(userName.empty() || password.empty());
    if (AlienImGui::Button("Create user")) {
        close();
        _createUserDialog->open(userName, password, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(userName.empty());
    if (AlienImGui::Button("Reset password")) {
        close();
        _resetPasswordDialog->open(userName, LoginController::get().getUserInfo());
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienImGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}
