#include "ModalWindow.h"

#include "AlienGui.h"
#include "StyleRepository.h"
#include "WindowController.h"

ModalWindow::ModalWindow(std::string const& title, RealVector2D const& defaultSize, bool maximizable)
    : _title(title)
    , _defaultSize(defaultSize)
    , _isMaximizable(maximizable)
{}

void ModalWindow::open()
{
    _state = State::JustOpened;
}

void ModalWindow::close()
{
    ImGui::CloseCurrentPopup();
    _state = State::Closed;
}

bool ModalWindow::isOpen() const
{
    return _state != State::Closed;
}

void ModalWindow::setTitle(std::string const& title)
{
    _title = title;
}

void ModalWindow::process(std::function<void()> const& contentFunc)
{
    if (_state == State::Closed) {
        return;
    }
    if (_state == State::JustOpened) {
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize({scale(_defaultSize.x), scale(_defaultSize.y)}, ImGuiCond_FirstUseEver);
        ImGui::OpenPopup(_title.c_str());
        _state = State::Open;
        _windowState = WindowState::Normal;
    }
    auto& style = ImGui::GetStyle();
    auto origWindowMinSize = style.WindowMinSize;
    style.WindowMinSize.x = scale(350.0f);
    style.WindowMinSize.y = scale(150.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_None;
    if (_isMaximizable && _windowState == WindowState::Maximized) {
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
        contentFunc();
        ImGui::PopID();

        ImGui::EndPopup();
    } else if (_state == State::Open) {
        // The popup can also be closed from outside, for instance by pressing escape
        _state = State::Closed;
    }

    style.WindowMinSize = origWindowMinSize;
}

void ModalWindow::processMaximizeButton()
{
    auto titlebarHeight = ImGui::GetFrameHeight();
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();
    auto iconSize = ImGui::GetFontSize();
    auto iconPos = ImVec2(windowPos.x + windowSize.x - scale(24.0f), windowPos.y + (titlebarHeight - iconSize) * 0.5f);

    // BeginPopupModal() clips widgets to the area below its native titlebar, so widen the clip rect
    // to make the button visible and clickable inside that titlebar strip.
    ImGui::PushClipRect(windowPos, ImVec2(windowPos.x + windowSize.x, windowPos.y + titlebarHeight), false);
    auto clicked = AlienGui::MaximizeButton(iconPos, iconSize, _windowState == WindowState::Maximized);
    ImGui::PopClipRect();

    if (clicked) {
        if (_windowState == WindowState::Maximized) {
            ImGui::SetWindowPos(_savedPos);
            ImGui::SetWindowSize(_savedSize);
            _windowState = WindowState::Normal;
        } else {
            _savedPos = windowPos;
            _savedSize = windowSize;
            _windowState = WindowState::Maximized;
        }
    }
}
