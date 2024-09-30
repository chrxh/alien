#include <imgui.h>

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

OverlayMessageController& OverlayMessageController::getInstance()
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
    processSpinner();

    if (!_show) {
        return;
    }
    processMessage();
}

void OverlayMessageController::show(std::string const& message, bool withLightning /*= false*/)
{
    _show = true;
    _message = message;
    _startTimepoint = std::chrono::steady_clock::now();
    _withLightning = withLightning;
    _counter = 0;
}

void OverlayMessageController::setOn(bool value)
{
    _on = value;
}

void OverlayMessageController::processSpinner()
{
    if (_persisterController->isBusy()) {
        if (!_spinnerRefTimepoint.has_value()) {
            _spinnerRefTimepoint = std::chrono::steady_clock::now();
        }
        auto now = std::chrono::steady_clock::now();
        auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - *_spinnerRefTimepoint).count());

        ImDrawList* drawList = ImGui::GetForegroundDrawList();

        AlienImGui::RotateStart(drawList);
        auto font = StyleRepository::getInstance().getIconFont();
        auto text = ICON_FA_SPINNER;
        ImVec4 clipRect(-100000.0f, -100000.0f, 100000.0f, 100000.0f);
        auto viewSize = toRealVector2D(Viewport::getViewSize());
        font->RenderText(
            drawList,
            scale(30.0f),
            {viewSize.x / 2 - scale(15.0f), viewSize.y - scale(80.0f)},
            ImColor::HSV(0.5f, 0.1f, 1.0f, std::min(1.0f, duration / 500)),
            clipRect,
            text,
            text + strlen(text),
            0.0f,
            false);

        auto angle = std::sinf(duration * Const::DegToRad / 10 - Const::Pi / 2) * 4 + 6.0f;
        _spinnerAngle += angle;
        AlienImGui::RotateEnd(_spinnerAngle, drawList);
    } else {
        _spinnerAngle = 0;
        _spinnerRefTimepoint.reset();
    }
}

void OverlayMessageController::processMessage()
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_startTimepoint);
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
