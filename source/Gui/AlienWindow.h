#pragma once

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"

#include "Definitions.h"
#include "StyleRepository.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "WindowController.h"
#include "Viewport.h"

template<typename ...Dependencies>
class AlienWindow : public MainLoopEntity<Dependencies...>
{
public:
    AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn, bool maximizable = false);

    bool isOn() const;
    void setOn(bool value);

protected:
    virtual void initIntern(Dependencies... dependencies) {}
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
    void init(Dependencies... dependencies) override;
    void process() override;
    void shutdown() override;

    std::string _title;
    std::optional<float> _cachedTitleWidth;

    bool _isMaximizable = false;
    enum class WindowState
    {
        Normal,
        Maximized,
        Collapsed
    };
    WindowState _state = WindowState::Normal;
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

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename ... Dependencies>
AlienWindow<Dependencies...>::AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn, bool maximizable)
    : _title(title)
    , _settingsNode(settingsNode)
    , _defaultOn(defaultOn)
    , _isMaximizable(maximizable)
{
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::init(Dependencies... dependencies)
{
    _on = GlobalSettings::get().getValue(_settingsNode + ".active", _defaultOn);
    initIntern(dependencies...);
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::process()
{
    processBackground();

    if (!isShown()) {
        return;
    }
    ImGui::PushID(_title.c_str());

    _savedWindowMinSize = ImGui::GetStyle().WindowMinSize;

    auto flags = returnFlagsAndConfigureNextWindow();

    if (ImGui::Begin(_title.c_str(), nullptr, flags)) {
        if (_state == WindowState::Normal) {
            _savedPos = ImGui::GetWindowPos();
            _savedSize = ImGui::GetWindowSize();
        }
        _isFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

        processTitlebar();

        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }
        if (_state != WindowState::Collapsed) {
            if (ImGui::BeginChild("child")) {
                processIntern();
            }
            ImGui::EndChild();
        }
    }
    ImGui::End();

    ImGui::GetStyle().WindowMinSize.y = _savedWindowMinSize.y;

    ImGui::PopID();
}

template <typename ... Dependencies>
bool AlienWindow<Dependencies...>::isOn() const
{
    return _on;
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::setOn(bool value)
{
    _on = value;
    if (value) {
        processActivated();
    }
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::shutdown()
{
    shutdownIntern();
    GlobalSettings::get().setValue(_settingsNode + ".active", _on);
}

template <typename ... Dependencies>
ImGuiWindowFlags AlienWindow<Dependencies...>::returnFlagsAndConfigureNextWindow()
{
    if (!_cachedTitleWidth.has_value()) {
        _cachedTitleWidth = ImGui::CalcTextSize(_title.c_str()).x;
    }

    if (_state == WindowState::Maximized) {
        ImGui::SetNextWindowBgAlpha(Const::MaximizedWindowAlpha * ImGui::GetStyle().Alpha);
        ImGui::SetNextWindowPos({0, ImGui::GetFrameHeight()}, ImGuiCond_Always);
        auto size = toRealVector2D(Viewport::get().getViewSize());
        ImGui::SetNextWindowSize({size.x, size.y - ImGui::GetFrameHeight()}, ImGuiCond_Always);
        return ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
    } else if (_state == WindowState::Collapsed) {
        ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
        ImGui::GetStyle().WindowMinSize.x = _cachedTitleWidth.value() + scale(100);
        ImGui::GetStyle().WindowMinSize.y = ImGui::GetTextLineHeightWithSpacing() + 1.0f;
        auto titlebarHeight = ImGui::GetTextLineHeightWithSpacing();
        ImGui::SetNextWindowSize({_savedSize.x, titlebarHeight}, ImGuiCond_Always);
        return ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
    } else {
        ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
        ImGui::SetNextWindowSize({scale(650.0f), scale(350.0f)}, ImGuiCond_FirstUseEver);
        ImGui::GetStyle().WindowMinSize.x = _cachedTitleWidth.value() + scale(100);
        return ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
    }
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::processTitlebar()
{
    drawTitlebarBackground();
    drawTitle();
    processCollapseButton();
    processMaximizeButton();
    processCloseButton();

    auto windowPos = ImGui::GetWindowPos();
    auto titlebarHeight = ImGui::GetTextLineHeightWithSpacing();
    auto const& framePadding = ImGui::GetStyle().FramePadding;
    ImGui::SetCursorScreenPos({windowPos.x + framePadding.x * 2, windowPos.y + titlebarHeight + framePadding.y * 2});
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::drawTitlebarBackground()
{
    auto titlebarHeight = ImGui::GetTextLineHeightWithSpacing();
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();

    auto bgColor = _state == WindowState::Collapsed || !_isFocused ? ImGui::GetColorU32(ImGuiCol_TitleBgCollapsed) : ImGui::GetColorU32(ImGuiCol_TitleBgActive);
    auto rounding = ImGui::GetStyle().WindowRounding;
    ImGui::GetWindowDrawList()->AddRectFilled(
        windowPos,
        ImVec2(windowPos.x + windowSize.x, windowPos.y + titlebarHeight + 1.0f),
        bgColor,
        rounding,
        ImDrawFlags_RoundCornersTopLeft | ImDrawFlags_RoundCornersTopRight);
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::drawTitle()
{
    auto windowPos = ImGui::GetWindowPos();
    auto iconSize = ImGui::GetFontSize();
    ImGui::SetCursorScreenPos(ImVec2(windowPos.x + iconSize + ImGui::GetStyle().FramePadding.x * 3 + 1, windowPos.y + ImGui::GetStyle().FramePadding.y));
    ImGui::TextUnformatted(_title.c_str());
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::processCollapseButton()
{
    auto titlebarHeight = ImGui::GetTextLineHeightWithSpacing();
    auto windowPos = ImGui::GetWindowPos();
    auto iconSize = ImGui::GetFontSize();
    auto iconPos = ImVec2(windowPos.x + 4, windowPos.y + (titlebarHeight - iconSize) * 0.5f);
    auto iconCenter = ImVec2(iconPos.x + iconSize * 0.5f, iconPos.y + iconSize * 0.5f);

    // Process interaction field
    ImGui::SetCursorScreenPos(iconPos);
    if (ImGui::InvisibleButton("CollapseButton", ImVec2(iconSize, iconSize))) {
        if (_state == WindowState::Collapsed) {
            ImGui::SetWindowSize(_savedSize);
            _state = WindowState::Normal;
        } else {
            if (_state == WindowState::Maximized) {
                ImGui::SetWindowPos(_savedPos);
                ImGui::SetWindowSize(_savedSize);
            }
            _state = WindowState::Collapsed;
        }
    }

    // Draw background circle
    auto pressed = ImGui::IsItemActive();
    bool hovered = ImGui::IsItemHovered();
    auto drawList = ImGui::GetWindowDrawList();
    auto center = ImVec2(iconPos.x + iconSize * 0.5f, iconPos.y + iconSize * 0.5f);
    if (hovered || pressed) {
        auto bgColor = hovered && pressed ? ImGui::GetColorU32(ImGuiCol_ButtonActive) : ImGui::GetColorU32(ImGuiCol_ButtonHovered);
        auto radius = iconSize * 0.6f;
        drawList->AddCircleFilled(iconCenter, radius, bgColor, 12);
    }

    // Draw icon
    {
        auto color = ImGui::GetColorU32(ImGuiCol_Text);
        auto radius = iconSize * 0.35f;
        if (_state == WindowState::Collapsed) {
            // Triangle pointing right
            drawList->AddTriangleFilled(
                ImVec2(center.x - radius * 0.5f, center.y - radius),
                ImVec2(center.x - radius * 0.5f, center.y + radius),
                ImVec2(center.x + radius, center.y),
                color);
        } else {
            // Triangle pointing down
            drawList->AddTriangleFilled(
                ImVec2(center.x - radius, center.y - radius * 0.5f),
                ImVec2(center.x + radius, center.y - radius * 0.5f),
                ImVec2(center.x, center.y + radius),
                color);
        }
    }
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::processMaximizeButton()
{
    if (!_isMaximizable) {
        return;
    }

    auto titlebarHeight = ImGui::GetTextLineHeightWithSpacing();
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();
    auto iconSize = ImGui::GetFontSize();
    auto iconPos = ImVec2(windowPos.x + windowSize.x - scale(24.0f) * 2, windowPos.y + (titlebarHeight - iconSize) * 0.5f);
    auto iconCenter = ImVec2(iconPos.x + iconSize * 0.5f, iconPos.y + iconSize * 0.5f);

    // Process interaction field
    ImGui::SetCursorScreenPos(iconPos);
    if (ImGui::InvisibleButton("MaximizeButton", ImVec2(iconSize, iconSize))) {
        if (_state == WindowState::Maximized) {
            ImGui::SetWindowPos(_savedPos);
            ImGui::SetWindowSize(_savedSize);
            _state = WindowState::Normal;
        } else {
            _state = WindowState::Maximized;
        }
    }

    // Draw background circle
    auto pressed = ImGui::IsItemActive();
    bool hovered = ImGui::IsItemHovered();
    auto drawList = ImGui::GetWindowDrawList();
    auto center = ImVec2(iconPos.x + iconSize * 0.5f, iconPos.y + iconSize * 0.5f);
    if (hovered || pressed) {
        auto bgColor = hovered && pressed ? ImGui::GetColorU32(ImGuiCol_ButtonActive) : ImGui::GetColorU32(ImGuiCol_ButtonHovered);
        auto radius = iconSize * 0.6f;
        drawList->AddCircleFilled(iconCenter, radius, bgColor, 12);
    }

    // Draw icon
    {
        if (_state == WindowState::Maximized) {
            drawList->AddText(
                StyleRepository::get().getIconFont(),
                iconSize * 0.7f,
                ImVec2(center.x - iconSize * 0.31f, center.y - iconSize * 0.22f),
                ImGui::GetColorU32(ImGuiCol_Text),
                ICON_FA_COMPRESS_ARROWS_ALT);
        } else {
            drawList->AddText(
                StyleRepository::get().getIconFont(),
                iconSize * 0.7f,
                ImVec2(center.x - iconSize * 0.31f, center.y - iconSize * 0.22f),
                ImGui::GetColorU32(ImGuiCol_Text),
                ICON_FA_EXPAND_ARROWS_ALT);
        }
    }
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::processCloseButton()
{
    auto titlebarHeight = ImGui::GetTextLineHeightWithSpacing();
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();
    auto iconSize = ImGui::GetFontSize();
    auto iconPos = ImVec2(windowPos.x + windowSize.x - scale(24.0f), windowPos.y + (titlebarHeight - iconSize) * 0.5f);
    auto iconCenter = ImVec2(iconPos.x + iconSize * 0.5f, iconPos.y + iconSize * 0.5f);

    // Process interaction field
    ImGui::SetCursorScreenPos(iconPos);
    if (ImGui::InvisibleButton("CloseButton", ImVec2(iconSize, iconSize))) {
        _on = false;
    }

    // Draw background circle
    auto pressed = ImGui::IsItemActive();
    bool hovered = ImGui::IsItemHovered();
    auto drawList = ImGui::GetWindowDrawList();
    auto center = ImVec2(iconPos.x + iconSize * 0.5f, iconPos.y + iconSize * 0.5f);
    if (hovered || pressed) {
        auto bgColor = hovered && pressed ? ImGui::GetColorU32(ImGuiCol_ButtonActive) : ImGui::GetColorU32(ImGuiCol_ButtonHovered);
        auto radius = iconSize * 0.6f;
        drawList->AddCircleFilled(iconCenter, radius, bgColor, 12);
    }

    // Draw cross
    {
        auto iconColor = ImGui::GetColorU32(ImGuiCol_Text);
        auto radius = iconSize * 0.3f;
        auto thickness = scale(1.0f);
        drawList->AddLine(ImVec2(center.x - radius, center.y - radius), ImVec2(center.x + radius, center.y + radius), iconColor, thickness);
        drawList->AddLine(ImVec2(center.x + radius, center.y - radius), ImVec2(center.x - radius, center.y + radius), iconColor, thickness);
    }
}
