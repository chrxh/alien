#include "GenericMessageDialog.h"

#include <boost/algorithm/string.hpp>

#include <imgui.h>

#include "Base/LoggingService.h"

#include "AlienImGui.h"
#include "WindowController.h"
#include "StyleRepository.h"

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
    changeTitle(title);
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
    changeTitle(title);
}

GenericMessageDialog::GenericMessageDialog()
    : AlienDialog("Message")
{
}

void GenericMessageDialog::processInformation()
{
    if (!_sizeInitialized) {
        auto size = ImGui::GetWindowSize();
        auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
        ImGui::SetWindowSize({size.x * factor, size.y * factor});
        _sizeInitialized = true;
    }

    ImGui::TextWrapped(_message.c_str());

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        close();
    }
}

void GenericMessageDialog::processYesNo()
{
    if (!_sizeInitialized) {
        auto size = ImGui::GetWindowSize();
        auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
        ImGui::SetWindowSize({size.x * factor, size.y * factor});
        _sizeInitialized = true;
    }

    ImGui::TextWrapped(_message.c_str());

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("Yes")) {
        close();
        _execFunction();
    }
    ImGui::SameLine();
    if (AlienImGui::Button("No")) {
        close();
    }
}
