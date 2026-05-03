#include "OverlayController.h"

#include <cmath>
#include <tuple>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>

#include <PersisterInterface/PersisterFacade.h>
#include "AlienGui.h"
#include "MainLoopEntityController.h"
#include "StyleRepository.h"
#include "UiController.h"
#include "Viewport.h"

namespace
{
    auto constexpr MessageFontSize = 48.0f;
    auto constexpr ShowDuration = 800;
    auto constexpr FadeoutTextDuration = 800;
    auto constexpr FadeoutLightningDuration = 1200;
    auto constexpr FadeoutProgressAnimationDuration = 1000;
}

void OverlayController::setup() {}

void OverlayController::process()
{
    if (!UiController::get().isOn()) {
        return;
    }
    if (!_on) {
        return;
    }
    processProgressAnimation();

    if (!_messageStartTimepoint.has_value()) {
        return;
    }
    processMessage();
}

void OverlayController::showMessage(std::string const& message, bool withLightning /*= false*/)
{
    _message = message;
    _messageStartTimepoint = std::chrono::steady_clock::now();
    _withLightning = withLightning;
    _counter = 0;
}

bool OverlayController::isOn() const
{
    return _on;
}

void OverlayController::setOn(bool value)
{
    _on = value;
}

void OverlayController::activateProgressAnimation(bool value) {}

void OverlayController::processProgressAnimation()
{
    if (_PersisterFacade::get()->isBusy()) {
        _busyTimepoint = std::chrono::steady_clock::now();
    }
    if (!_busyTimepoint.has_value()) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    auto millisecSinceBusy = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_busyTimepoint).count();

    if (millisecSinceBusy < FadeoutProgressAnimationDuration) {
        if (!_progressBarRefTimepoint.has_value()) {
            _progressBarRefTimepoint = std::chrono::steady_clock::now();
        }
        auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - *_progressBarRefTimepoint).count());
        auto alpha = 1.0f - toFloat(millisecSinceBusy) / FadeoutProgressAnimationDuration;

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto viewSize = toRealVector2D(Viewport::get().getViewSize());
        auto width = scale(300.0f);
        auto height = scale(15.0f);
        auto center = ImVec2{viewSize.x / 2, viewSize.y - scale(60.0f)};

        drawList->AddRectFilledMultiColor(
            ImVec2{center.x, center.y - height * 5},
            ImVec2{center.x - width, center.y + height},
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 1.0f * alpha));
        drawList->AddRectFilledMultiColor(
            ImVec2{center.x, center.y - height * 5},
            ImVec2{center.x + width, center.y + height},
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 1.0f * alpha));
        drawList->AddRectFilledMultiColor(
            ImVec2{center.x + width, center.y + height},
            ImVec2{center.x, center.y + height * 6},
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 1.0f * alpha),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f));
        drawList->AddRectFilledMultiColor(
            ImVec2{center.x - width, center.y + height},
            ImVec2{center.x, center.y + height * 6},
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 1.0f * alpha),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f),
            ImColor::HSV(0.66f, 1.0f, 0.1f, 0.0f));
        auto const helixSegments = 64;
        auto const helixTurns = 2.75f;
        auto const helixRadius = scale(22.0f);
        auto const helixWidth = width * 1.15f;
        auto const maxDepth = sqrtf(helixWidth * helixWidth / 4 + helixRadius * helixRadius);
        auto const perspectiveDistance = maxDepth * 3.5f;
        auto const phase = duration / 420.0f + 1.4f * sinf(duration / 700.0f);
        auto const yRotationPhase = duration / 1800.0f;
        auto const maxYRotationAngle = 50.0f * Const::Pi / 180.0f;
        auto const yRotationAngle = sinf(yRotationPhase) * maxYRotationAngle;
        auto const strandThickness = scale(2.0f);
        auto const rungThickness = scale(1.0f);
        auto const dotRadius = scale(2.2f);

        auto getPoint = [&](int index, float offset) {
            auto t = toFloat(index) / toFloat(helixSegments - 1);
            auto angle = t * helixTurns * 2.0f * Const::Pi + phase + offset;
            auto xRotation = (t - 0.5f) * helixWidth;
            auto yRotation = sinf(angle) * helixRadius;
            auto zRotation = cosf(angle) * helixRadius;
            auto rotatedX = xRotation * cosf(yRotationAngle) + zRotation * sinf(yRotationAngle);
            auto rotatedZ = -xRotation * sinf(yRotationAngle) + zRotation * cosf(yRotationAngle);
            auto perspective = perspectiveDistance / (perspectiveDistance - rotatedZ);
            auto x = center.x + rotatedX * perspective;
            auto y = center.y + height / 2 + yRotation;
            auto depth = (rotatedZ / maxDepth + 1.0f) / 2.0f;
            return std::tuple<ImVec2, float>{ImVec2{x, y}, depth};
        };

        for (int i = 0; i < helixSegments; i += 2) {
            auto [point1, depth1] = getPoint(i, 0.0f);
            auto [point2, depth2] = getPoint(i, Const::Pi);
            auto depth = (depth1 + depth2) / 2.0f;
            drawList->AddLine(point1, point2, ImColor::HSV(0.58f, 0.7f, 0.5f, (0.25f + depth * 0.25f) * alpha), rungThickness);
        }

        for (int i = 1; i < helixSegments; ++i) {
            auto [previousPoint1, previousDepth1] = getPoint(i - 1, 0.0f);
            auto [currentPoint1, currentDepth1] = getPoint(i, 0.0f);
            auto [previousPoint2, previousDepth2] = getPoint(i - 1, Const::Pi);
            auto [currentPoint2, currentDepth2] = getPoint(i, Const::Pi);
            auto depth1 = (previousDepth1 + currentDepth1) / 2.0f;
            auto depth2 = (previousDepth2 + currentDepth2) / 2.0f;

            drawList->AddLine(
                previousPoint1, currentPoint1, ImColor::HSV(0.61f, 0.85f, 0.65f + depth1 * 0.25f, (0.45f + depth1 * 0.35f) * alpha), strandThickness);
            drawList->AddLine(
                previousPoint2, currentPoint2, ImColor::HSV(0.67f, 0.85f, 0.65f + depth2 * 0.25f, (0.45f + depth2 * 0.35f) * alpha), strandThickness);
        }

        for (int i = 0; i < helixSegments; i += 2) {
            auto [point1, depth1] = getPoint(i, 0.0f);
            auto [point2, depth2] = getPoint(i, Const::Pi);
            drawList->AddCircleFilled(
                point1, dotRadius * (0.75f + depth1 * 0.5f), ImColor::HSV(0.61f, 0.75f, 0.75f + depth1 * 0.2f, (0.5f + depth1 * 0.4f) * alpha));
            drawList->AddCircleFilled(
                point2, dotRadius * (0.75f + depth2 * 0.5f), ImColor::HSV(0.67f, 0.75f, 0.75f + depth2 * 0.2f, (0.5f + depth2 * 0.4f) * alpha));
        }
        drawList->AddText(
            StyleRepository::get().getReefMediumFont(),
            scale(16.0f),
            {center.x - scale(28.0f), center.y - scale(15.0f)},
            ImColor::HSV(0, 0, 1, 0.7f * alpha),
            "Processing");


    } else {
        _progressBarRefTimepoint.reset();
        _busyTimepoint.reset();
    }
}

void OverlayController::processMessage()
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_messageStartTimepoint);
    if (duration.count() > ShowDuration + FadeoutTextDuration) {
        _messageStartTimepoint.reset();
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