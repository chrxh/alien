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
        auto flags = ImGuiWindowFlags_NoCollapse;
        shouldProcess = ImGui::Begin(_title.c_str(), NULL, flags);
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

        if (_isModal) {
            ImGui::EndPopup();
        } else {
            ImGui::End();
        }
    } else if (!_isModal) {
        // Non-modal window was closed by user clicking X button
        _state = DialogState::Closed;
    }

    style.WindowMinSize = origWindowMinSize;
}

void AlienDialog::shutdown()
{
    shutdownIntern();
}
