#include "AlienDialog.h"


AlienDialog::AlienDialog(std::string const& title, ImVec2 const& defaultSize)
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
        ImGui::OpenPopup(_title.c_str());
        _state = DialogState::Open;
    }
    auto& style = ImGui::GetStyle();
    auto origWindowMinSize = style.WindowMinSize;
    style.WindowMinSize.x = scale(350.0f);
    style.WindowMinSize.y = scale(150.0f);

    if (ImGui::BeginPopupModal(_title.c_str(), NULL, 0)) {
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
    }

    style.WindowMinSize = origWindowMinSize;
}

void AlienDialog::shutdown()
{
    shutdownIntern();
}
