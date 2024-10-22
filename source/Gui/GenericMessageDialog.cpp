#include "GenericMessageDialog.h"

#include <boost/algorithm/string.hpp>

#include <imgui.h>

#include "Base/LoggingService.h"

#include "AlienImGui.h"
#include "WindowController.h"

void GenericMessageDialog::processIntern()
{
    switch (_dialogType) {
    case DialogType::Information:
        processInformation();
        break;
    case DialogType::YesNo:
        processYesNo();
        break;
    }
}

void GenericMessageDialog::information(std::string const& title, std::string const& message)
{
    _title = title;
    _message = message;
    _dialogType = DialogType::Information;
    log(Priority::Important, "message dialog showing: '" + message + "'");

    AlienDialog::open();
    changeTitle(title + "##msg");
}

void GenericMessageDialog::information(std::string const& title, std::vector<PersisterErrorInfo> const& errors)
{
    std::vector<std::string> errorMessages;
    for (auto const& error : errors) {
        errorMessages.emplace_back(error.message);
    }
    GenericMessageDialog::get().information(title, boost::join(errorMessages, "\n\n"));
}

void GenericMessageDialog::yesNo(std::string const& title, std::string const& message, std::function<void()> const& yesFunction)
{
    _title = title;
    _message = message;
    _dialogType = DialogType::YesNo;
    _execFunction = yesFunction;

    AlienDialog::open();
    changeTitle(title + "##msg");
}

GenericMessageDialog::GenericMessageDialog()
    : AlienDialog("Message")
{
}

void GenericMessageDialog::processInformation()
{
    ImGui::OpenPopup((_title + "##msg").c_str());
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal((_title + "##msg").c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }

        AlienImGui::Text(_message);
        AlienImGui::Separator();

        if (AlienImGui::Button("OK")) {
            close();
        }
        ImGui::EndPopup();
    }
}

void GenericMessageDialog::processYesNo()
{
    ImGui::OpenPopup(_title.c_str());
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(_title.c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }

        AlienImGui::Text(_message);
        AlienImGui::Separator();

        if (AlienImGui::Button("Yes")) {
            close();
            _execFunction();
        }
        ImGui::SameLine();
        if (AlienImGui::Button("No")) {
            close();
        }
        ImGui::EndPopup();
    }
}
