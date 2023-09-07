#include <imgui.h>

#include "AlienDialog.h"
#include "StyleRepository.h"
#include "WindowController.h"

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
            auto& windowController = WindowController::getInstance();
            auto factor = windowController.getContentScaleFactor() / windowController.getLastContentScaleFactor();
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
    ImGui::CloseCurrentPopup();
    _state = DialogState::Closed;
}

void _AlienDialog::changeTitle(std::string const& title)
{
    _title = title;
}
