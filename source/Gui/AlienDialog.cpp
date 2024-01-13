#include <imgui.h>

#include "AlienDialog.h"
#include "StyleRepository.h"
#include "WindowController.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"

_AlienDialog::_AlienDialog(std::string const& title)
    : _title(title)
{
}

_AlienDialog::~_AlienDialog()
{
}

void _AlienDialog::process()
{
    if (_state == DialogState::Closed) {
        return;
    }
    if (_state == DialogState::JustOpened) {
        ImGui::OpenPopup(_title.c_str());
        _state = DialogState::Open;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(_title.c_str(), NULL, 0)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::getContentScaleFactor() / WindowController::getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }

        ImGui::PushID(_title.c_str());
        processIntern();
        ImGui::PopID();

        ImGui::EndPopup();
    }
}

void _AlienDialog::open()
{
    _state = DialogState::JustOpened;
    openIntern();
}

void _AlienDialog::close()
{
    delayedExecution([this] {
        ImGui::CloseCurrentPopup();
        _state = DialogState::Closed;
    });
    printOverlayMessage("Exiting ...");
}

void _AlienDialog::changeTitle(std::string const& title)
{
    _title = title;
}
