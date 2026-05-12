#include "AlienDialog.h"


AlienDialog::AlienDialog(std::string const& title, RealVector2D const& defaultSize)
    : _title(title)
    , _defaultSize(defaultSize)
{}

void AlienDialog::init()
{
    initIntern();
}

void AlienDialog::open()
{
    _state = DialogState::JustOpened;
    _isModal = true;
    openIntern();
}

void AlienDialog::open(bool asNonModal)
{
    _state = DialogState::JustOpened;
    _isModal = !asNonModal;
    openIntern();
}

void AlienDialog::close()
{
    delayedExecution([this] {
        ImGui::CloseCurrentPopup();
        _state = DialogState::Closed;
    });
}

void AlienDialog::changeTitle(std::string const& title)
{
    _title = title;
}

void AlienDialog::process()
{
    if (_state == DialogState::Closed) {
        return;
    }
    if (_state == DialogState::JustOpened) {
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize({scale(_defaultSize.x), scale(_defaultSize.y)}, ImGuiCond_FirstUseEver);
        if (_isModal) {
            ImGui::OpenPopup(_title.c_str());
        } else {
            // For non-modal dialogs opened from modals, bring to front
            ImGui::SetNextWindowFocus();
        }
        _state = DialogState::Open;
    }
    auto& style = ImGui::GetStyle();
    auto origWindowMinSize = style.WindowMinSize;
    style.WindowMinSize.x = scale(350.0f);
    style.WindowMinSize.y = scale(150.0f);

    bool shouldProcess = false;
    if (_isModal) {
        shouldProcess = ImGui::BeginPopupModal(_title.c_str(), NULL, 0);
    } else {
        // Use a popup window instead of a regular window to render above modals
        if (_state == DialogState::Open) {
            ImGui::OpenPopup(_title.c_str());
        }
        auto flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        shouldProcess = ImGui::BeginPopup(_title.c_str(), flags);
    }

    if (shouldProcess) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }


        ImGui::PushID(_title.c_str());
        processIntern();
        ImGui::PopID();

        ImGui::EndPopup();
    } else if (!_isModal && _state == DialogState::Open) {
        // Non-modal popup was closed
        _state = DialogState::Closed;
    }

    style.WindowMinSize = origWindowMinSize;
}

void AlienDialog::shutdown()
{
    shutdownIntern();
}
