#pragma once

#include <imgui.h>

#include <Base/GlobalSettings.h>

#include "Definitions.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "WindowController.h"

class AlienWindow : public MainLoopEntity
{
public:
    AlienWindow(
        std::string const& title,
        std::string const& settingsNode,
        bool defaultOn,
        bool maximizable = false,
        RealVector2D const& defaultPos = {60.0f, 60.0f},
        RealVector2D const& defaultSize = {650.0f, 350.0f},
        RealVector2D const& minSize = {300.0f, 100.0f});

    bool isOn() const;
    void setOn(bool value);

protected:
    virtual void initIntern() {}
    virtual void shutdownIntern() {}
    virtual void processIntern() = 0;
    virtual void processBackground() {}
    virtual void processActivated() {}

    virtual bool isShown() { return _on; }

    bool _on = false;
    bool _defaultOn = false;
    std::string _settingsNode;

private:
    void init() override;
    void process() override;
    void shutdown() override;

    std::string _title;

    bool _isMaximizable = false;
    enum class WindowState
    {
        Normal,
        Maximized,
        Collapsed
    };
    WindowState _state = WindowState::Normal;
    RealVector2D _defaultPos;
    RealVector2D _defaultSize;
    RealVector2D _minSize;
    bool _isFocused = false;
    RealVector2D _savedPos;
    RealVector2D _savedSize;
    RealVector2D _savedWindowMinSize;

    ImGuiWindowFlags returnFlagsAndConfigureNextWindow();

    void processTitlebar();

    void drawTitlebarBackground();
    void drawTitle();
    void processCollapseButton();
    void processMaximizeButton();
    void processCloseButton();
};
