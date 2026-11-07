#include "OverlayController.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <variant>
#include <vector>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>

#include <Persister/Interface/PersisterFacade.h>
#include "AlienGui.h"
#include "MainLoopEntityController.h"
#include "StyleService.h"
#include "UiController.h"
#include "Viewport.h"

namespace
{
    auto constexpr MessageFontSize = 48.0f;
    auto constexpr ShowDuration = 800;
    auto constexpr FadeoutTextDuration = 800;
    auto constexpr FadeoutLightningDuration = 1200;
    auto constexpr FadeoutProgressAnimationDuration = 1000;

    auto constexpr ScrimSegments = 64;
    auto constexpr ScrimCenterAlpha = 0.92f;
    auto constexpr HelixSegments = 160;
    auto constexpr HelixRungs = 30;
    auto constexpr HelixTurns = 3.0f;
    auto constexpr HelixWidth = 345.0f;
    auto constexpr HelixRadius = 21.0f;
    auto constexpr HelixPerspectiveFactor = 3.5f;
    auto constexpr HelixTilt = 0.32f;
    auto constexpr HelixRadiansPerMillisecond = 1.0f / 620.0f;

    struct ScrimRing
    {
        float extent;
        float alpha;
    };
    ScrimRing constexpr ScrimRings[] = {{0.45f, 0.8f}, {0.75f, 0.38f}, {1.0f, 0.0f}};

    struct HelixStrand
    {
        float angleOffset;
        float hue;
    };
    HelixStrand constexpr HelixStrands[] = {{0.0f, 0.61f}, {Const::Pi, 0.79f}};

    struct HelixPoint
    {
        ImVec2 pos;
        float depth;
    };

    struct HelixLine
    {
        ImVec2 start;
        ImVec2 end;
        float thickness;
    };

    struct HelixDot
    {
        ImVec2 pos;
        float radius;
    };

    struct HelixPrimitive
    {
        std::variant<HelixLine, HelixDot> shape;
        float depth;
        ImU32 color;
    };
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

    if (_messageStartTimepoint.has_value()) {
        processMessage();
    } else if (_status.has_value()) {
        drawMessage(*_status, 1.0f);
    }
}

void OverlayController::showMessage(std::string const& message, bool withLightning /*= false*/)
{
    _message = message;
    _messageStartTimepoint = std::chrono::steady_clock::now();
    _withLightning = withLightning;
    _counter = 0;
}

void OverlayController::setStatus(std::optional<std::string> const& status)
{
    _status = status;
}

bool OverlayController::isOn() const
{
    return _on;
}

void OverlayController::setOn(bool value)
{
    _on = value;
}

namespace
{
    void drawScrim(ImDrawList* drawList, ImVec2 const& center, float alpha)
    {
        auto numRings = static_cast<int>(std::size(ScrimRings));
        auto whitePixel = ImGui::GetFontTexUvWhitePixel();
        auto getColor = [&](float scrimAlpha) { return ImColor::HSV(0.66f, 1.0f, 0.1f, scrimAlpha * alpha); };

        drawList->PrimReserve(ScrimSegments * 3 + (numRings - 1) * ScrimSegments * 6, 1 + numRings * ScrimSegments);
        auto centerIndex = drawList->_VtxCurrentIdx;
        drawList->PrimWriteVtx(center, whitePixel, getColor(ScrimCenterAlpha));
        for (auto const& ring : ScrimRings) {
            for (auto segment : std::views::iota(0, ScrimSegments)) {
                auto angle = toFloat(segment) / toFloat(ScrimSegments) * 2.0f * Const::Pi;
                auto pos = ImVec2{center.x + cosf(angle) * scale(290.0f) * ring.extent, center.y + sinf(angle) * scale(99.0f) * ring.extent};
                drawList->PrimWriteVtx(pos, whitePixel, getColor(ring.alpha));
            }
        }

        auto getIndex = [&](int ring, int segment) { return static_cast<ImDrawIdx>(centerIndex + 1 + ring * ScrimSegments + segment % ScrimSegments); };
        for (auto segment : std::views::iota(0, ScrimSegments)) {
            drawList->PrimWriteIdx(static_cast<ImDrawIdx>(centerIndex));
            drawList->PrimWriteIdx(getIndex(0, segment));
            drawList->PrimWriteIdx(getIndex(0, segment + 1));
            for (auto ring : std::views::iota(1, numRings)) {
                drawList->PrimWriteIdx(getIndex(ring - 1, segment));
                drawList->PrimWriteIdx(getIndex(ring, segment));
                drawList->PrimWriteIdx(getIndex(ring, segment + 1));
                drawList->PrimWriteIdx(getIndex(ring - 1, segment));
                drawList->PrimWriteIdx(getIndex(ring, segment + 1));
                drawList->PrimWriteIdx(getIndex(ring - 1, segment + 1));
            }
        }
    }

    void drawHelix(ImDrawList* drawList, ImVec2 const& center, float animationTime, float alpha)
    {
        auto width = scale(HelixWidth);
        auto radius = scale(HelixRadius);
        auto perspectiveDistance = sqrtf(width * width / 4 + radius * radius) * HelixPerspectiveFactor;
        auto phase = animationTime * HelixRadiansPerMillisecond;

        auto project = [&](float x, float y, float z) {
            auto tiltedX = x * cosf(HelixTilt) + z * sinf(HelixTilt);
            auto tiltedZ = -x * sinf(HelixTilt) + z * cosf(HelixTilt);
            auto perspective = perspectiveDistance / (perspectiveDistance - tiltedZ);
            return ImVec2{tiltedX * perspective, y * perspective};
        };
        auto offsetX = (project(-width / 2, 0.0f, 0.0f).x + project(width / 2, 0.0f, 0.0f).x) / 2;

        auto getPoint = [&](float t, float angleOffset) {
            auto angle = t * HelixTurns * 2.0f * Const::Pi + phase + angleOffset;
            auto projected = project((t - 0.5f) * width, sinf(angle) * radius, cosf(angle) * radius);
            return HelixPoint{{center.x + projected.x - offsetX, center.y + projected.y}, (cosf(angle) + 1.0f) / 2.0f};
        };

        std::vector<HelixPrimitive> primitives;
        primitives.reserve(std::size(HelixStrands) * (HelixSegments - 1 + HelixRungs) + HelixRungs);
        for (auto const& strand : HelixStrands) {
            for (auto index : std::views::iota(1, HelixSegments)) {
                auto start = getPoint(toFloat(index - 1) / toFloat(HelixSegments - 1), strand.angleOffset);
                auto end = getPoint(toFloat(index) / toFloat(HelixSegments - 1), strand.angleOffset);
                auto depth = (start.depth + end.depth) / 2;
                auto color = ImColor::HSV(strand.hue, 0.72f - depth * 0.2f, 0.34f + depth * 0.66f, alpha);
                primitives.emplace_back(HelixPrimitive{HelixLine{start.pos, end.pos, scale(1.5f + depth * 1.2f)}, depth, color});
            }
            for (auto index : std::views::iota(0, HelixRungs)) {
                auto point = getPoint(toFloat(index) / toFloat(HelixRungs - 1), strand.angleOffset);
                auto color = ImColor::HSV(strand.hue, 0.4f, 0.4f + point.depth * 0.6f, alpha);
                primitives.emplace_back(HelixPrimitive{HelixDot{point.pos, scale(1.4f + point.depth * 1.5f)}, point.depth + 0.01f, color});
            }
        }
        for (auto index : std::views::iota(0, HelixRungs)) {
            auto t = toFloat(index) / toFloat(HelixRungs - 1);
            auto color = ImColor::HSV(0.71f, 0.5f, 0.62f, 0.5f * alpha);
            primitives.emplace_back(HelixPrimitive{HelixLine{getPoint(t, 0.0f).pos, getPoint(t, Const::Pi).pos, scale(1.0f)}, 0.5f, color});
        }
        std::ranges::sort(primitives, {}, &HelixPrimitive::depth);

        for (auto const& primitive : primitives) {
            if (auto line = std::get_if<HelixLine>(&primitive.shape)) {
                drawList->AddLine(line->start, line->end, primitive.color, line->thickness);
            } else if (auto dot = std::get_if<HelixDot>(&primitive.shape)) {
                drawList->AddCircleFilled(dot->pos, dot->radius, primitive.color);
            }
        }
    }
}

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
        auto center = ImVec2{viewSize.x / 2, viewSize.y - scale(66.0f)};

        drawScrim(drawList, {center.x, center.y + scale(8.0f)}, alpha);
        drawHelix(drawList, center, duration, alpha);

        auto font = StyleService::get().getReefMediumFont();
        auto fontSize = scale(16.0f);
        auto text = "Processing";
        auto textWidth = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text).x;
        drawList->AddText(font, fontSize, {center.x - textWidth / 2, center.y + scale(32.0f)}, ImColor::HSV(0, 0, 1, 0.82f * alpha), text);
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

    if (_withLightning && ++_counter > 2) {
        auto durationAfterSomeTicks = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_ticksLaterTimepoint);
        float lightningAlpha = std::max(0.0f, 0.7f - static_cast<float>(durationAfterSomeTicks.count()) / FadeoutLightningDuration);
        auto viewSize = toRealVector2D(Viewport::get().getViewSize());
        ImGui::GetForegroundDrawList()->AddRectFilled({0, 0}, {viewSize.x, viewSize.y}, ImColor::HSV(0.0f, 0.0f, 1.0f, lightningAlpha));
    }
    drawMessage(_message, textAlpha);
}

void OverlayController::drawMessage(std::string const& message, float alpha)
{
    ImDrawList* drawList = ImGui::GetForegroundDrawList();

    auto& styleRep = StyleService::get();
    auto center = ImGui::GetMainViewport()->Size;
    center.x /= 2;
    auto textColorFront = ImColor::HSV(0.5f, 0.0f, 1.0f, alpha);
    auto textColorBack = ImColor::HSV(0.5f, 0.0f, 0.0f, alpha);

    ImGui::PushFont(styleRep.getMonospaceLargeFont());
    auto fontScaling = MessageFontSize / styleRep.getMonospaceLargeFont()->FontSize;
    auto fontSize = ImGui::CalcTextSize(message.c_str());
    fontSize.x *= fontScaling;
    fontSize.y *= fontScaling;
    ImGui::PopFont();

    drawList->AddText(
        styleRep.getMonospaceLargeFont(),
        styleRep.scale(MessageFontSize),
        {center.x - fontSize.x / 2 - 2, center.y - styleRep.scale(100.0f) - fontSize.y - 2},
        textColorBack,
        message.c_str());

    drawList->AddText(
        styleRep.getMonospaceLargeFont(),
        styleRep.scale(MessageFontSize),
        {center.x - fontSize.x / 2, center.y - styleRep.scale(100.0f) - fontSize.y},
        textColorFront,
        message.c_str());
}