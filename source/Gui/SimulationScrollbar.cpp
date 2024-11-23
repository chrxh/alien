#include "SimulationScrollbar.h"

#include <algorithm>

#include <imgui.h>

#include "EngineInterface/SimulationFacade.h"
#include "Viewport.h"
#include "StyleRepository.h"

_SimulationScrollbar::_SimulationScrollbar(
    std::string const& id,
    Orientation orientation,
    SimulationFacade const& simulationFacade)
    : _id(id)
    , _orientation(orientation)
    , _simulationFacade(simulationFacade)
{}

void _SimulationScrollbar::process(RealRect const& rect)
{
    processEvents(rect);

    auto size = rect.bottomRight - rect.topLeft;
    auto sliderbarRect = calcSliderbarRect(rect);
    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration;

    ImGui::SetNextWindowPos(ImVec2(rect.topLeft.x, rect.topLeft.y));
    ImGui::SetNextWindowSize(ImVec2(size.x, size.y));
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin(_id.c_str(), NULL, windowFlags);

    ImColor sliderColor = doesMouseCursorIntersectSliderBar(rect) ? ImColor(Const::SimulationSliderColor_Active)
                                                                  : ImColor(Const::SimulationSliderColor_Base);
    sliderColor.Value.w *= ImGui::GetStyle().Alpha;
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(rect.topLeft.x + sliderbarRect.topLeft.x, rect.topLeft.y + sliderbarRect.topLeft.y),
        ImVec2(rect.topLeft.x + sliderbarRect.bottomRight.x, rect.topLeft.y + sliderbarRect.bottomRight.y),
        sliderColor,
        5.0f);

    ImGui::End();
    ImGui::PopStyleVar();
}

void _SimulationScrollbar::processEvents(RealRect const& rect)
{
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (doesMouseCursorIntersectSliderBar(rect)) {
            _worldCenterForDragging = Viewport::get().getCenterInWorldPos();
        }
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && _worldCenterForDragging) {
        auto dragViewDelta = ImGui::GetMouseDragDelta();
        auto scrollbarSize = rect.bottomRight - rect.topLeft;
        auto worldSize = _simulationFacade->getWorldSize();
        auto dragWorldDelta = RealVector2D{
            dragViewDelta.x / scrollbarSize.x * worldSize.x, dragViewDelta.y / scrollbarSize.y * worldSize.y};
        auto centerInWorldPos = Viewport::get().getCenterInWorldPos();
        if (Orientation::Horizontal == _orientation) {
            centerInWorldPos.x = _worldCenterForDragging->x + dragWorldDelta.x;
        } else {
            centerInWorldPos.y = _worldCenterForDragging->y + dragWorldDelta.y;
        }
        Viewport::get().setCenterInWorldPos(centerInWorldPos);
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        _worldCenterForDragging = std::nullopt;
    }
}

RealRect _SimulationScrollbar::calcSliderbarRect(RealRect const& scrollbarRect) const
{
    auto size2d = scrollbarRect.bottomRight - scrollbarRect.topLeft;
    auto worldSize =
        Orientation::Horizontal == _orientation ? _simulationFacade->getWorldSize().x : _simulationFacade->getWorldSize().y;
    auto size = Orientation::Horizontal == _orientation ? size2d.x : size2d.y;

    auto worldRect = Viewport::get().getVisibleWorldRect();
    auto startWorldPos = Orientation::Horizontal == _orientation ? worldRect.topLeft.x : worldRect.topLeft.y;
    auto endWorldPos = Orientation::Horizontal == _orientation ? worldRect.bottomRight.x : worldRect.bottomRight.y;

    auto sliderBarStartPos = std::min(std::max(startWorldPos / worldSize * size, 0.0f), size);
    auto sliderBarEndPos = std::min(std::max(endWorldPos / worldSize * size, 0.0f), size);
    if (sliderBarEndPos < sliderBarStartPos) {
        sliderBarEndPos = sliderBarStartPos;
    }
    auto sliderBarPos =
        Orientation::Horizontal == _orientation ? ImVec2{4 + sliderBarStartPos, 4} : ImVec2{4, 4 + sliderBarStartPos};
    auto sliderBarSize = Orientation::Horizontal == _orientation ? ImVec2{sliderBarEndPos - sliderBarStartPos - 8, 10}
                                                                 : ImVec2{10, sliderBarEndPos - sliderBarStartPos - 8};

    sliderBarSize = {std::max(10.0f, sliderBarSize.x), std::max(10.0f, sliderBarSize.y)};

    return {
        {sliderBarPos.x, sliderBarPos.y}, {sliderBarPos.x + sliderBarSize.x - 1, sliderBarPos.y + sliderBarSize.y - 1}};
}

bool _SimulationScrollbar::doesMouseCursorIntersectSliderBar(RealRect const& rect) const
{
    auto sliderbarRect = calcSliderbarRect(rect);

    ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
    return mousePositionAbsolute.x >= rect.topLeft.x + sliderbarRect.topLeft.x - 3
        && mousePositionAbsolute.x <= rect.topLeft.x + sliderbarRect.bottomRight.x + 3
        && mousePositionAbsolute.y >= rect.topLeft.y + sliderbarRect.topLeft.y - 3
        && mousePositionAbsolute.y <= rect.topLeft.y + sliderbarRect.bottomRight.y + 3;
}
