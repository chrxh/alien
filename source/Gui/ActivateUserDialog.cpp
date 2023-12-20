#include "ActivateUserDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "StyleRepository.h"

_ActivateUserDialog::_ActivateUserDialog(
    SimulationController const& simController,
    BrowserWindow const& browserWindow,
    NetworkService const& networkController)
    : _AlienDialog("Activate user")
    , _simController(simController)
    , _browserWindow(browserWindow)
    , _networkService(networkController)
{}

_ActivateUserDialog::~_ActivateUserDialog() {}

void _ActivateUserDialog::registerCyclicReferences(CreateUserDialogWeakPtr const& createUserDialog)
{
    _createUserDialog = createUserDialog;
}


void _ActivateUserDialog::open(std::string const& userName, std::string const& password, UserInfo const& userInfo)
{
    _AlienDialog::open();
    _userName = userName;
    _password = password;
    _userInfo = userInfo;
}

void _ActivateUserDialog::processIntern()
{
    AlienImGui::Text("Please enter the confirmation code sent to your email address.");
    AlienImGui::HelpMarker(
        "Please check your spam folder if you did not find an email. If you did not receive an email there, try signing up with possibly another "
        "email address. If this still does not work, please contact info@alien-project.org.");
    AlienImGui::Separator();
    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Code (case sensitive)").textWidth(0), _confirmationCode);

    AlienImGui::Separator();

    ImGui::BeginDisabled(_confirmationCode.empty());
    if (AlienImGui::Button("OK")) {
        close();
        onActivateUser();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    AlienImGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienImGui::Button("Resend")) {
        _createUserDialog.lock()->onCreateUser();
    }

    ImGui::SameLine();
    if (AlienImGui::Button("Resend to other email address")) {
        close();
        _createUserDialog.lock()->open(_userName, _password, _userInfo);
    }

    ImGui::SameLine();
    AlienImGui::VerticalSeparator();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void _ActivateUserDialog::onActivateUser()
{
    auto result = _networkService->activateUser(_userName, _password, _userInfo, _confirmationCode);
    if (result) {
        LoginErrorCode errorCode;
        result |= _networkService->login(errorCode, _userName, _password, _userInfo);
    }
    if (!result) {
        MessageDialog::getInstance().information("Error", "An error occurred on the server. Your entered code may be incorrect.\nPlease try to register again.");
    } else {
        MessageDialog::getInstance().information(
            "Information",
            "The user '" + _userName
                + "' has been successfully created.\nYou are logged in and are now able to upload your own simulations\nor upvote others by likes.");
        _browserWindow->onRefresh();
    }
}
