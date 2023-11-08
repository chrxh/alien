#include <imgui.h>

#include "AlienImGui.h"
#include "OverlayMessageController.h"

#include "StyleRepository.h"

namespace
{
    auto constexpr MessageFontSize = 48.0f;
    auto constexpr ShowDuration = 800;
    auto constexpr FadeoutDuration = 800;
}

OverlayMessageController& OverlayMessageController::getInstance()
{
    static OverlayMessageController instance;
    return instance;
}

void OverlayMessageController::process()
{
    if (!_show) {
        return;
    }
    auto now= std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_startTimePoint);
    if (duration.count() > ShowDuration + FadeoutDuration) {
        _show = false;
    }

    float alpha = 1.0f;
    if (duration.count() > ShowDuration) {
        alpha = std::max(0.0f, 1.0f - static_cast<float>(duration.count() - ShowDuration) / FadeoutDuration);
    }

    ImDrawList* drawList = ImGui::GetForegroundDrawList();

    auto styleRep = StyleRepository::getInstance();
    auto center = ImGui::GetMainViewport()->Size;
    center.x /= 2;
    auto textColorFront = ImColor::HSV(0.5f, 0.0f, 1.0f, alpha);
    auto textColorBack = ImColor::HSV(0.5f, 0.0f, 0.0f, alpha);

    ImGui::PushFont(styleRep.getMonospaceLargeFont());
    auto fontScaling = MessageFontSize / styleRep.getMonospaceLargeFont()->FontSize;
    auto fontSize = ImGui::CalcTextSize(_message.c_str());
    fontSize.x *= fontScaling;
    fontSize.y *= fontScaling;
    ImGui::PopFont();

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

void OverlayMessageController::show(std::string const& message)
{
    _show = true;
    _message = message;
    _startTimePoint = std::chrono::steady_clock::now();
}
