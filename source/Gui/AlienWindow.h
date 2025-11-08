#pragma once

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

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
        RealVector2D const& minSize = {scale(300.0f), scale(100.0f)});

    bool isOn() const;
    void setOn(bool value);

protected:
    virtual void initIntern() {}
    virtual void shutdownIntern() {}
    virtual void processIntern() = 0;
    virtual void processBackground() {}
    virtual void processActivated() {}

    virtual bool isShown() { return _on; }

    bool _sizeInitialized = false;
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
    RealVector2D _minSize;
    bool _isFocused = false;
    ImVec2 _savedPos;
    ImVec2 _savedSize;
    ImVec2 _savedWindowMinSize;

    ImGuiWindowFlags returnFlagsAndConfigureNextWindow();

    void processTitlebar();

    void drawTitlebarBackground();
    void drawTitle();
    void processCollapseButton();
    void processMaximizeButton();
    void processCloseButton();
};
