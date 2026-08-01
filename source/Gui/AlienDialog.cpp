#include "AlienDialog.h"

#include "AlienGui.h"

AlienDialog::AlienDialog(std::string const& title, RealVector2D const& defaultSize, bool maximizable)
    : _title(title)
    , _defaultSize(defaultSize)
    , _isMaximizable(maximizable)
{}

void AlienDialog::init()
{
    initIntern();
}

void AlienDialog::open()
{
    _nested = false;
    _state = DialogState::JustOpened;
    openIntern();
}

void AlienDialog::openNested()
{
    _nested = true;
    _state = DialogState::JustOpened;
    openIntern();
}

void AlienDialog::processNested()
{
    if (_nested) {
        processDialog();
    }
}

bool AlienDialog::isOpen() const
{
    return _state != DialogState::Closed;
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
    if (_nested) {
        return;
    }
    processDialog();
}

void AlienDialog::processDialog()
{
    if (_state == DialogState::Closed) {
        return;
    }
    if (_state == DialogState::JustOpened) {
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize({scale(_defaultSize.x), scale(_defaultSize.y)}, ImGuiCond_FirstUseEver);
        ImGui::OpenPopup(_title.c_str());
        _state = DialogState::Open;
        _windowState = DialogWindowState::Normal;
    }
    auto& style = ImGui::GetStyle();
    auto origWindowMinSize = style.WindowMinSize;
    style.WindowMinSize.x = scale(350.0f);
    style.WindowMinSize.y = scale(150.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_None;
    if (_isMaximizable && _windowState == DialogWindowState::Maximized) {
        auto viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(viewport->Size, ImGuiCond_Always);
        flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    }

    if (ImGui::BeginPopupModal(_title.c_str(), NULL, flags)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }

        if (_isMaximizable) {
            processMaximizeButton();
        }

        ImGui::PushID(_title.c_str());
        processIntern();
        ImGui::PopID();

        ImGui::EndPopup();
    } else if (_state == DialogState::Open) {
        // The popup can also be closed from outside, for instance by pressing escape
        _state = DialogState::Closed;
    }

    style.WindowMinSize = origWindowMinSize;
}

void AlienDialog::processMaximizeButton()
{
    auto titlebarHeight = ImGui::GetFrameHeight();
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();
    auto iconSize = ImGui::GetFontSize();
    auto iconPos = ImVec2(windowPos.x + windowSize.x - scale(24.0f), windowPos.y + (titlebarHeight - iconSize) * 0.5f);

    // BeginPopupModal() clips widgets to the area below its native titlebar, so widen the clip rect
    // to make the button visible and clickable inside that titlebar strip.
    ImGui::PushClipRect(windowPos, ImVec2(windowPos.x + windowSize.x, windowPos.y + titlebarHeight), false);
    auto clicked = AlienGui::MaximizeButton(iconPos, iconSize, _windowState == DialogWindowState::Maximized);
    ImGui::PopClipRect();

    if (clicked) {
        if (_windowState == DialogWindowState::Maximized) {
            ImGui::SetWindowPos(_savedPos);
            ImGui::SetWindowSize(_savedSize);
            _windowState = DialogWindowState::Normal;
        } else {
            _savedPos = windowPos;
            _savedSize = windowSize;
            _windowState = DialogWindowState::Maximized;
        }
    }
}

void AlienDialog::shutdown()
{
    shutdownIntern();
}
