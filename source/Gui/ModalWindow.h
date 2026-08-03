#pragma once

#include <functional>
#include <string>

#include <imgui.h>

#include "Definitions.h"

// Modal popup window whose content is drawn by a callback. In contrast to AlienDialog it holds no content state and is
// no main loop entity, therefore it can also be owned by a widget and processed within its draw pass.
class ModalWindow
{
public:
    ModalWindow(std::string const& title, RealVector2D const& defaultSize = RealVector2D(450.0f, 150.0f), bool maximizable = false);

    void open();

    // Needs to be called while the window is being processed, since the underlying popup is closed immediately
    void close();

    bool isOpen() const;
    void setTitle(std::string const& title);

    // Draws the modal window and calls contentFunc inside of it
    void process(std::function<void()> const& contentFunc);

private:
    void processMaximizeButton();

    std::string _title;
    RealVector2D _defaultSize;
    bool _isMaximizable = false;

    bool _sizeInitialized = false;
    enum class State
    {
        Closed,
        JustOpened,
        Open
    };
    State _state = State::Closed;

    enum class WindowState
    {
        Normal,
        Maximized
    };
    WindowState _windowState = WindowState::Normal;
    ImVec2 _savedPos;
    ImVec2 _savedSize;
};
