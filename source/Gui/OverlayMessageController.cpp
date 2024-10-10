#include <imgui.h>
#include <cmath>

#include "AlienImGui.h"
#include "OverlayMessageController.h"

#include <Fonts/IconsFontAwesome5.h>

#include "StyleRepository.h"
#include "Viewport.h"
#include "Base/Math.h"

namespace
{
    auto constexpr MessageFontSize = 48.0f;
    auto constexpr ShowDuration = 800;
    auto constexpr FadeoutTextDuration = 800;
    auto constexpr FadeoutLightningDuration = 1200;
}

OverlayMessageController& OverlayMessageController::get()
{
    static OverlayMessageController instance;
    return instance;
}

void OverlayMessageController::init(PersisterController const& persisterController)
{
    _persisterController = persisterController;
}

void OverlayMessageController::process()
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

void OverlayMessageController::show(std::string const& message, bool withLightning /*= false*/)
{
    _show = true;
    _message = message;
    _messageStartTimepoint = std::chrono::steady_clock::now();
    _withLightning = withLightning;
    _counter = 0;
}

void OverlayMessageController::setOn(bool value)
{
    _on = value;
}

void OverlayMessageController::processLoadingBar()
{
    if (_persisterController->isBusy()) {
        if (!_progressBarRefTimepoint.has_value()) {
            _progressBarRefTimepoint = std::chrono::steady_clock::now();
        }
        auto now = std::chrono::steady_clock::now();
        auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - *_progressBarRefTimepoint).count());

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto viewSize = toRealVector2D(Viewport::getViewSize());
        auto width = viewSize.x / 6 + 1.0f;
        auto height = scale(20.0f);
        auto center = ImVec2{viewSize.x / 2, viewSize.y - scale(60.0f)};
        drawList->AddRectFilled(
            ImVec2{center.x - width / 2, center.y - height / 2}, ImVec2{center.x + width / 2, center.y + height / 2},
            ImColor::HSV(0.66f, 1.0f, 0.5f, 0.4f));

        auto progressWidth = width * 2 / 5;
        auto progressStart = (toInt(duration) / 10 + toInt(width - progressWidth)) % toInt(width);
        auto progressEnd = (toInt(duration) / 10) % toInt(width);
        if (progressStart < progressEnd) {
            drawList->AddRectFilledMultiColor(
                ImVec2{center.x - width / 2 + toFloat(progressStart), center.y - height / 2},
                ImVec2{center.x - width / 2 + toFloat(progressEnd), center.y + height / 2},
                ImColor::HSV(0.66f, 0.9f, 1.0f, 1.0f),
                ImColor::HSV(0.66f, 0.9f, 0.6f, 1.0f),
                ImColor::HSV(0.66f, 0.9f, 0.2f, 1.0f),
                ImColor::HSV(0.66f, 0.9f, 0.6f, 1.0f));
        } else {
            {
                auto factor = toFloat(progressEnd) / progressWidth;
                auto brightness1 = 1.0f * factor + 0.6f * (1.0f - factor);
                auto brightness2 = 0.6f * factor + 0.2f * (1.0f - factor);
                drawList->AddRectFilledMultiColor(
                    ImVec2{center.x - width / 2, center.y - height / 2},
                    ImVec2{center.x - width / 2 + toFloat(progressEnd), center.y + height / 2},
                    ImColor::HSV(0.66f, 0.9f, brightness1, 1.0f),
                    ImColor::HSV(0.66f, 0.9f, 0.6f, 1.0f),
                    ImColor::HSV(0.66f, 0.9f, 0.2f, 1.0f),
                    ImColor::HSV(0.66f, 0.9f, brightness2, 1.0f));
            }
            {
                auto factor = (width - toFloat(progressStart)) / progressWidth;
                auto brightness3 = 1.0f * (1.0f - factor) + 0.6f * factor;
                auto brightness4 = 0.6f * (1.0f - factor) + 0.2f * factor;
                drawList->AddRectFilledMultiColor(
                    ImVec2{center.x - width / 2 + toFloat(progressStart), center.y - height / 2},
                    ImVec2{center.x + width / 2, center.y + height / 2},
                    ImColor::HSV(0.66f, 0.9f, 1.0f, 1.0f),
                    ImColor::HSV(0.66f, 0.9f, brightness3, 1.0f),
                    ImColor::HSV(0.66f, 0.9f, brightness4, 1.0f),
                    ImColor::HSV(0.66f, 0.9f, 0.6f, 1.0f));
            }
        }
        drawList->AddRect(
            ImVec2{center.x - width / 2, center.y - height / 2}, ImVec2{center.x + width / 2, center.y + height / 2}, ImColor::HSV(0.66f, 0.7f, 0.8f, 1.0f));

    } else {
        _progressBarRefTimepoint.reset();
    }
}

void OverlayMessageController::processMessage()
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
        auto durationAfterSomeTicks = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_ticksLaterTimepoint);
        float lightningAlpha = std::max(0.0f, 0.7f - static_cast<float>(durationAfterSomeTicks.count()) / FadeoutLightningDuration);
        auto viewSize = toRealVector2D(Viewport::getViewSize());
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
