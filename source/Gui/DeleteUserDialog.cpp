#include "DeleteUserDialog.h"

#include <imgui.h>

#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "CreateUserDialog.h"
#include "MessageDialog.h"

_DeleteUserDialog::_DeleteUserDialog(BrowserWindow const& browserWindow)
    : _AlienDialog("Delete user")
    , _browserWindow(browserWindow)
{
}

void _DeleteUserDialog::processIntern()
{
    AlienImGui::Text(
        "Warning: All the data of the user '" + *NetworkService::getLoggedInUserName()
        + "' will be deleted on the server side.\nThese include the likes, the simulations and the account data.");
    AlienImGui::Separator();

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Re-enter password").password(true).textWidth(0), _reenteredPassword);
    AlienImGui::Separator();

    ImGui::BeginDisabled(_reenteredPassword.empty());
    if (AlienImGui::Button("Delete")) {
        close();
        if (_reenteredPassword == *NetworkService::getPassword()) {
            onDelete();
        } else {
            MessageDialog::getInstance().information("Error", "The password does not match.");
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
    auto userName = *NetworkService::getLoggedInUserName();
    if (NetworkService::deleteUser()) {
        _browserWindow->onRefresh();
        MessageDialog::getInstance().information("Information", "The user '" + userName + "' has been deleted.\nYou are logged out.");
    } else {
        MessageDialog::getInstance().information("Error", "An error occurred on the server.");
    }
}
