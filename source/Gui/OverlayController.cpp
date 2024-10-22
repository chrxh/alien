#include <imgui.h>
#include <cmath>

#include "AlienImGui.h"
#include "OverlayController.h"

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"

#include "MainLoopEntityController.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr MessageFontSize = 48.0f;
    auto constexpr ShowDuration = 800;
    auto constexpr FadeoutTextDuration = 800;
    auto constexpr FadeoutLightningDuration = 1200;
}

void OverlayController::setup(PersisterFacade const& persisterFacade)
{
    _persisterFacade = persisterFacade;
}

void OverlayController::process()
{
    if (!_on) {
        return;
    }
    processLoadingBar();

    if (!_show) {
        return;
    }
    processMessage();
}

void OverlayController::showMessage(std::string const& message, bool withLightning /*= false*/)
{
    _show = true;
    _message = message;
    _messageStartTimepoint = std::chrono::steady_clock::now();
    _withLightning = withLightning;
    _counter = 0;
}

void OverlayController::setOn(bool value)
{
    _on = value;
}

void OverlayController::processLoadingBar()
{
    if (_persisterFacade->isBusy()) {
        if (!_progressBarRefTimepoint.has_value()) {
            _progressBarRefTimepoint = std::chrono::steady_clock::now();
        }
        auto now = std::chrono::steady_clock::now();
        auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - *_progressBarRefTimepoint).count());

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto viewSize = toRealVector2D(Viewport::get().getViewSize());
        auto width = viewSize.x / 6 + 1.0f;
        auto height = scale(20.0f);
        auto center = ImVec2{viewSize.x / 2, viewSize.y - scale(60.0f)};

        auto constexpr N = 20;
        for (int i = 0; i < N; ++i) {
            auto amplitude1 = sinf(toFloat(i) * 10.0f / toFloat(N) - duration / 240.0f);
            auto amplitude2 = sinf(toFloat(i) * 14.0f / toFloat(N) - duration / 135.0f);
            //auto hue = toFloat((i * 1000 / N + toInt(duration)) % 3000) / 4500.0f;
            //hue = hue < 0.33f ? 0.66f + hue : 0.66f + 0.66f - hue; 

            drawList->AddRectFilled(
                ImVec2{center.x - width / 2 + toFloat(i) / N * width, center.y + height / 2 - amplitude1 * height},
                ImVec2{center.x - width / 2 + toFloat(i + 1) / N * width - scale(3), center.y + height / 2 - amplitude2 * height / 2},
                ImColor::HSV(0, 0.1f, 0.35f, 0.6f));
        }
        drawList->AddText(
            StyleRepository::get().getReefMediumFont(),
            scale(16.0f),
            {center.x - scale(28.0f), center.y - scale(15.0f)},
            ImColor::HSV(0, 0, 1, 0.7f),
            "Processing");


    } else {
        _progressBarRefTimepoint.reset();
    }
}

void OverlayController::processMessage()
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_messageStartTimepoint);
    if (duration.count() > ShowDuration + FadeoutTextDuration) {
        _show = false;
    }
    if (_counter == 2) {
        _ticksLaterTimepoint = std::chrono::steady_clock::now();
    }

    float textAlpha = 1.0f;
    if (duration.count() > ShowDuration) {
        textAlpha = std::max(0.0f, 1.0f - static_cast<float>(duration.count() - ShowDuration) / FadeoutTextDuration);
    }
    ImDrawList* drawList = ImGui::GetForegroundDrawList();

    auto& styleRep = StyleRepository::get();
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
        auto durationAfterSomeTicks = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_ticksLaterTimepoint);
        float lightningAlpha = std::max(0.0f, 0.7f - static_cast<float>(durationAfterSomeTicks.count()) / FadeoutLightningDuration);
        auto viewSize = toRealVector2D(Viewport::get().getViewSize());
        drawList->AddRectFilled({0, 0}, {viewSize.x, viewSize.y}, ImColor::HSV(0.0f, 0.0f, 1.0f, lightningAlpha));
    }
    drawList->AddText(
        styleRep.getMonospaceLargeFont(),
        styleRep.scale(MessageFontSize),
        {center.x - fontSize.x / 2 - 2, center.y - styleRep.scale(100.0f) - fontSize.y - 2},
        textColorBack,
        _message.c_str());

    drawList->AddText(
        styleRep.getMonospaceLargeFont(),
        styleRep.scale(MessageFontSize),
        {center.x - fontSize.x / 2, center.y - styleRep.scale(100.0f) - fontSize.y},
        textColorFront,
        _message.c_str());
}
