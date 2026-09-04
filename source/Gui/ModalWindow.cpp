#include "ModalWindow.h"

#include <Base/GlobalSettings.h>

#include "AlienGui.h"
#include "StyleRepository.h"
#include "WindowController.h"

ModalWindow::ModalWindow(std::string const& title, RealVector2D const& defaultSize, bool maximizable, std::string const& settingsNode)
    : _title(title)
    , _defaultSize{scale(defaultSize.x), scale(defaultSize.y)}
    , _isMaximizable(maximizable)
    , _settingsNode(settingsNode)
{}

void ModalWindow::open()
{
    loadSettings();
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
        ImGui::SetNextWindowSize({_defaultSize.x, _defaultSize.y}, ImGuiCond_FirstUseEver);
        ImGui::OpenPopup(_title.c_str());
        _state = State::Open;
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
    auto iconPos = RealVector2D{windowPos.x + windowSize.x - scale(24.0f), windowPos.y + (titlebarHeight - iconSize) * 0.5f};

    // BeginPopupModal() clips widgets to the area below its native titlebar, so widen the clip rect
    // to make the button visible and clickable inside that titlebar strip.
    ImGui::PushClipRect(windowPos, ImVec2(windowPos.x + windowSize.x, windowPos.y + titlebarHeight), false);
    auto clicked = AlienGui::MaximizeButton(iconPos, iconSize, _windowState == WindowState::Maximized);
    ImGui::PopClipRect();

    if (clicked) {
        if (_windowState == WindowState::Maximized) {
            ImGui::SetWindowPos({_savedPos.x, _savedPos.y});
            ImGui::SetWindowSize({_savedSize.x, _savedSize.y});
            _windowState = WindowState::Normal;
        } else {
            _savedPos = {windowPos.x, windowPos.y};
            _savedSize = {windowSize.x, windowSize.y};
            _windowState = WindowState::Maximized;
        }
        saveSettings();
    }
}

void ModalWindow::loadSettings()
{
    if (_settingsNode.empty()) {
        return;
    }

    auto& settings = GlobalSettings::get();
    _windowState = static_cast<WindowState>(settings.getValue(_settingsNode + ".state", toInt(_windowState)));
    _savedPos.x = settings.getValue(_settingsNode + ".saved pos.x", _savedPos.x);
    _savedPos.y = settings.getValue(_settingsNode + ".saved pos.y", _savedPos.y);
    auto correction = WindowController::get().getContentScaleCorrection();
    _savedSize.x = settings.getValue(_settingsNode + ".saved size.x", _savedSize.x) * correction;
    _savedSize.y = settings.getValue(_settingsNode + ".saved size.y", _savedSize.y) * correction;
}

void ModalWindow::saveSettings()
{
    if (_settingsNode.empty()) {
        return;
    }

    auto& settings = GlobalSettings::get();
    settings.setValue(_settingsNode + ".state", toInt(_windowState));
    settings.setValue(_settingsNode + ".saved pos.x", _savedPos.x);
    settings.setValue(_settingsNode + ".saved pos.y", _savedPos.y);
    settings.setValue(_settingsNode + ".saved size.x", _savedSize.x);
    settings.setValue(_settingsNode + ".saved size.y", _savedSize.y);
}
