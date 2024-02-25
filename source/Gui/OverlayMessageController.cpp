#include <imgui.h>

#include "AlienImGui.h"
#include "OverlayMessageController.h"

#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr MessageFontSize = 48.0f;
    auto constexpr ShowDuration = 800;
    auto constexpr FadeoutTextDuration = 800;
    auto constexpr FadeoutLightningDuration = 1200;
}

OverlayMessageController& OverlayMessageController::getInstance()
{
    static OverlayMessageController instance;
    return instance;
}

void OverlayMessageController::process()
{
    if (!_show || !_on) {
        return;
    }
    auto now= std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_startTimePoint);
    if (duration.count() > ShowDuration + FadeoutTextDuration) {
        _show = false;
    }
    if (_counter == 2) {
        _ticksLaterTimePoint = std::chrono::steady_clock::now();
    }

    float textAlpha = 1.0f;
    if (duration.count() > ShowDuration) {
        textAlpha = std::max(0.0f, 1.0f - static_cast<float>(duration.count() - ShowDuration) / FadeoutTextDuration);
    }
    ImDrawList* drawList = ImGui::GetForegroundDrawList();

    auto& styleRep = StyleRepository::getInstance();
    auto center = ImGui::GetMainViewport()->Size;
    center.x /= 2;
    auto textColorFront = ImColor::HSV(0.5f, 0.0f, 1.0f, textAlpha);
    auto textColorBack = ImColor::HSV(0.5f, 0.0f, 0.0f, textAlpha);

    ImGui::PushFont(styleRep.getMonospaceLargeFont());
    auto fontScaling = MessageFontSize / styleRep.getMonospaceLargeFont()->FontSize;
    auto fontSize = ImGui::CalcTextSize(_message.c_str());
    fontSize.x *= fontScaling;
    fontSize.y *= fontScaling;
    ImGui::PopFont();

    if (_withLightning && ++_counter > 2) {
        auto durationAfterSomeTicks = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_ticksLaterTimePoint);
        float lightningAlpha = std::max(0.0f, 0.7f - static_cast<float>(durationAfterSomeTicks.count()) / FadeoutLightningDuration);
        auto viewSize = toRealVector2D(Viewport::getViewSize());
        drawList->AddRectFilled({0, 0}, {viewSize.x, viewSize.y}, ImColor::HSV(0.0f, 0.0f, 1.0f, lightningAlpha));
    }
    drawList->AddText(
        styleRep.getMonospaceLargeFont(),
        styleRep.scale(MessageFontSize),
        {center.x - fontSize.x / 2 - 2, center.y - styleRep.scale(50.0f) - fontSize.y - 2},
        textColorBack,
        _message.c_str());

    drawList->AddText(
        styleRep.getMonospaceLargeFont(),
        styleRep.scale(MessageFontSize),
        {center.x - fontSize.x / 2, center.y - styleRep.scale(50.0f) - fontSize.y},
        textColorFront,
        _message.c_str());
}

void OverlayMessageController::show(std::string const& message, bool withLightning /*= false*/)
{
    _show = true;
    _message = message;
    _startTimePoint = std::chrono::steady_clock::now();
    _withLightning = withLightning;
    _counter = 0;
}

void OverlayMessageController::setOn(bool value)
{
    _on = value;
}
