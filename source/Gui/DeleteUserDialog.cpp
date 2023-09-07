#include "DeleteUserDialog.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "MessageDialog.h"
#include "NetworkController.h"

_DeleteUserDialog::_DeleteUserDialog(BrowserWindow const& browserWindow, NetworkController const& networkController)
    : _AlienDialog("Delete user")
    , _browserWindow(browserWindow)
    , _networkController(networkController)
{
}

void _DeleteUserDialog::processIntern()
{
    AlienImGui::Text(
        "Warning: All the data of the user '" + *_networkController->getLoggedInUserName()
        + "' will be deleted on the server side.\nThese include the likes, the simulations and the account data.");
    AlienImGui::Separator();

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Re-enter password").password(true).textWidth(0), _reenteredPassword);
    AlienImGui::Separator();

    ImGui::BeginDisabled(_reenteredPassword.empty());
    if (AlienImGui::Button("Delete")) {
        close();
        if (_reenteredPassword == *_networkController->getPassword()) {
            onDelete();
        } else {
            MessageDialog::getInstance().show("Error", "The password does not match.");
        }
        _reenteredPassword.clear();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
        _reenteredPassword.clear();
    }
}

void _DeleteUserDialog::onDelete()
{
    auto userName = *_networkController->getLoggedInUserName();
    if (_networkController->deleteUser()) {
        _browserWindow->onRefresh();
        MessageDialog::getInstance().show("Information", "The user '" + userName + "' has been deleted.\nYou are logged out.");
    } else {
        MessageDialog::getInstance().show("Error", "An error occurred on the server.");
    }
}
