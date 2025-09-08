#include <algorithm>

#include <imgui.h>

#include "SimulationScrollbars.h"
#include "StyleRepository.h"

namespace 
{
    auto const ScrollbarThickness = 17.0f;
}

void _SimulationScrollbars::process(RealVector2D& worldCenter, RealRect const& worldRect, RealRect const& visibleWorldRect, RealRect const& viewRect)
{
    processScrollbar(worldCenter, worldRect, visibleWorldRect, viewRect, Orientation::Horizontal);
    processScrollbar(worldCenter, worldRect, visibleWorldRect, viewRect, Orientation::Vertical);
}

void _SimulationScrollbars::processScrollbar(
    RealVector2D& worldCenter,
    RealRect const& worldRect,
    RealRect const& visibleWorldRect,
    RealRect const& viewRect,
    Orientation orientation)
{
    auto scrollbarRect = calcScrollbarRect(viewRect, orientation);
    auto sliderbarRect = calcSliderbarRect(worldRect, visibleWorldRect, viewRect, orientation);  // Relative to scrollbarRect

    // Trim slider bar position to be within scrollbar rect (considering padding)
    if (orientation == Orientation::Horizontal) {
        if (sliderbarRect.bottomRight.x > scrollbarRect.bottomRight.x - scale(6.0f)) {
            sliderbarRect.bottomRight.x = scrollbarRect.bottomRight.x - scale(6.0f);
        }
    } else {
        if (sliderbarRect.bottomRight.y > scrollbarRect.bottomRight.y - scale(10.0f + ScrollbarThickness)) {
            sliderbarRect.bottomRight.y = scrollbarRect.bottomRight.y - scale(10.0f + ScrollbarThickness);
        }
    }

    auto size = scrollbarRect.bottomRight - scrollbarRect.topLeft;

    auto mouseCursorIntersectSliderbar = doesMouseCursorIntersectSliderBar(scrollbarRect, sliderbarRect);

    processEvents(worldCenter, worldRect, viewRect, mouseCursorIntersectSliderbar, orientation);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration;

    ImGui::SetNextWindowPos(ImVec2(scrollbarRect.topLeft.x, scrollbarRect.topLeft.y));
    ImGui::SetNextWindowSize(
        orientation == Orientation::Horizontal ? ImVec2(size.x, scale(ScrollbarThickness + 5.0f)) : ImVec2(scale(ScrollbarThickness + 5.0f), size.y));
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin(orientation == Orientation::Horizontal ? "horizontal" : "vertical", NULL, windowFlags);

    ImColor sliderColor = mouseCursorIntersectSliderbar ? ImColor(Const::SimulationSliderColor_Active) : ImColor(Const::SimulationSliderColor_Base);
    sliderColor.Value.w *= ImGui::GetStyle().Alpha;
    ImGui::GetForegroundDrawList()->AddRectFilled(
        ImVec2(scrollbarRect.topLeft.x + sliderbarRect.topLeft.x, scrollbarRect.topLeft.y + sliderbarRect.topLeft.y),
        ImVec2(scrollbarRect.topLeft.x + sliderbarRect.bottomRight.x,scrollbarRect.topLeft.y + sliderbarRect.bottomRight.y),
        sliderColor,
        5.0f);

    ImGui::End();
    ImGui::PopStyleVar();
}

RealRect _SimulationScrollbars::calcScrollbarRect(RealRect const& viewRect, Orientation orientation) const
{
    if (orientation == Orientation::Horizontal) {
        return RealRect{
            {viewRect.topLeft.x, viewRect.bottomRight.y - scale(ScrollbarThickness)},
            {viewRect.bottomRight.x - scale(1.0f + ScrollbarThickness), viewRect.bottomRight.y}};
    } else {
        return RealRect{
            {viewRect.bottomRight.x - scale(ScrollbarThickness), viewRect.topLeft.y},
            {viewRect.bottomRight.x, viewRect.bottomRight.y - scale(1.0f + ScrollbarThickness)}};
    }
}

void _SimulationScrollbars::processEvents(
    RealVector2D& worldCenter,
    RealRect const& worldRect,
    RealRect const& viewRect,
    bool mouseCursorIntersectSliderbar,
    Orientation orientation)
{
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (mouseCursorIntersectSliderbar) {
            _worldCenterForDragging = worldCenter;
            _orientationForDragging = orientation;
        }
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && _worldCenterForDragging && _orientationForDragging == orientation) {
        auto dragViewDelta = ImGui::GetMouseDragDelta();
        auto scrollbarSize = viewRect.bottomRight - viewRect.topLeft;
        auto worldSize = worldRect.bottomRight - worldRect.topLeft;
        auto dragWorldDelta = RealVector2D{dragViewDelta.x / scrollbarSize.x * worldSize.x, dragViewDelta.y / scrollbarSize.y * worldSize.y};
        auto newWorldCenter = worldCenter;
        if (Orientation::Horizontal == orientation) {
            newWorldCenter.x = _worldCenterForDragging->x + dragWorldDelta.x;
        } else {
            newWorldCenter.y = _worldCenterForDragging->y + dragWorldDelta.y;
        }
        worldCenter = newWorldCenter;
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        _worldCenterForDragging.reset();
        _orientationForDragging.reset();
    }
}

RealRect _SimulationScrollbars::calcSliderbarRect(RealRect const& worldRect, RealRect const& visibleWorldRect, RealRect const& viewRect, Orientation orientation) const
{
    auto size2d = viewRect.bottomRight - viewRect.topLeft;
    auto worldSize = Orientation::Horizontal == orientation ? worldRect.bottomRight.x - worldRect.topLeft.x : worldRect.bottomRight.y - worldRect.topLeft.y;
    auto size = Orientation::Horizontal == orientation ? size2d.x : size2d.y;

    auto startWorldPos = Orientation::Horizontal == orientation ? visibleWorldRect.topLeft.x : visibleWorldRect.topLeft.y;
    auto endWorldPos = Orientation::Horizontal == orientation ? visibleWorldRect.bottomRight.x : visibleWorldRect.bottomRight.y;

    auto sliderBarStartPos = std::min(std::max(startWorldPos / worldSize * size, 0.0f), size);
    auto sliderBarEndPos = std::min(std::max(endWorldPos / worldSize * size, 0.0f), size);
    if (sliderBarEndPos < sliderBarStartPos) {
        sliderBarEndPos = sliderBarStartPos;
    }
    auto sliderBarPos = Orientation::Horizontal == orientation ? ImVec2{scale(4) + sliderBarStartPos, scale(4)} : ImVec2{scale(4), scale(4) + sliderBarStartPos};
    auto sliderBarSize = Orientation::Horizontal == orientation ? ImVec2{sliderBarEndPos - sliderBarStartPos - scale(8), scale(10)}
                                                                : ImVec2{scale(10), sliderBarEndPos - sliderBarStartPos - scale(8)};

    sliderBarSize = {std::max(scale(10.0f), sliderBarSize.x), std::max(scale(10.0f), sliderBarSize.y)};

    return {{sliderBarPos.x, sliderBarPos.y}, {sliderBarPos.x + sliderBarSize.x - scale(1), sliderBarPos.y + sliderBarSize.y - scale(1)}};
}

bool _SimulationScrollbars::doesMouseCursorIntersectSliderBar(RealRect const& scrollbarRect, RealRect const& sliderbarRect) const
{
    ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
    auto extension = scale(3.0f);
    return mousePositionAbsolute.x >= scrollbarRect.topLeft.x + sliderbarRect.topLeft.x - extension
        && mousePositionAbsolute.x <= scrollbarRect.topLeft.x + sliderbarRect.bottomRight.x + extension
        && mousePositionAbsolute.y >= scrollbarRect.topLeft.y + sliderbarRect.topLeft.y - extension
        && mousePositionAbsolute.y <= scrollbarRect.topLeft.y + sliderbarRect.bottomRight.y + extension;
}
