#include "ScissorsController.h"

#include <ranges>

#include <Fonts/AlienIconFont.h>

#include <Base/GlobalSettings.h>
#include <Base/Math.h>

#include <EngineInterface/SimulationFacade.h>

#include "EditorModel.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr TrailDuration = 0.4f;
    auto constexpr TrailThickness = 3.0f;
    auto constexpr IconSize = 16.0f;
    auto constexpr IconOffset = 8.0f;
}

bool ScissorsController::isCutOnlyInSelection() const
{
    return _cutOnlyInSelection;
}

void ScissorsController::setCutOnlyInSelection(bool value)
{
    _cutOnlyInSelection = value;
}

void ScissorsController::onLeftMouseButtonPressed(RealVector2D const& viewPos)
{
    _trail.emplace_back(Viewport::get().mapViewToWorldPosition(viewPos), std::chrono::steady_clock::now(), true);
}

void ScissorsController::onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (ImGui::GetIO().KeyAlt || viewPos == prevViewPos) {
        return;
    }
    auto worldPos = Viewport::get().mapViewToWorldPosition(viewPos);
    cutConnections(Viewport::get().mapViewToWorldPosition(prevViewPos), worldPos);
    _trail.emplace_back(worldPos, std::chrono::steady_clock::now(), false);
}

void ScissorsController::drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const
{
    auto iconFont = StyleService::get().getIconFont();
    auto iconFontSize = scale(IconSize);
    auto iconPos = ImVec2{mousePos.x + scale(IconOffset), mousePos.y + scale(IconOffset)};
    drawList->AddText(iconFont, iconFontSize, {iconPos.x + scale(1.0f), iconPos.y + scale(1.0f)}, Const::CursorShadowColor, ICON_SCISSORS);
    drawList->AddText(iconFont, iconFontSize, iconPos, Const::ScissorsTrailColor, ICON_SCISSORS);
}

void ScissorsController::init()
{
    _cutOnlyInSelection = GlobalSettings::get().getValue("editors.scissors.only in selection", _cutOnlyInSelection);
}

void ScissorsController::process()
{
    auto now = std::chrono::steady_clock::now();
    auto age = [&](TrailPoint const& point) { return std::chrono::duration<float>(now - point.time).count(); };
    while (!_trail.empty() && age(_trail.front()) > TrailDuration) {
        _trail.pop_front();
    }
    if (!_trail.empty() && !_trail.front().startsSegment) {
        _trail.front().startsSegment = true;
    }

    auto drawList = ImGui::GetBackgroundDrawList();
    for (auto const& [from, to] : std::views::zip(_trail, _trail | std::views::drop(1))) {
        if (to.startsSegment) {
            continue;
        }
        auto color = Const::ScissorsTrailColor;
        color.Value.w *= 1.0f - std::min(1.0f, age(to) / TrailDuration);
        auto fromPos = Viewport::get().mapWorldToViewPosition(from.worldPos, false);
        auto toPos = Viewport::get().mapWorldToViewPosition(to.worldPos, false);
        drawList->AddLine({fromPos.x, fromPos.y}, {toPos.x, toPos.y}, color, scale(TrailThickness));
    }
}

void ScissorsController::shutdown()
{
    GlobalSettings::get().setValue("editors.scissors.only in selection", _cutOnlyInSelection);
}

void ScissorsController::cutConnections(RealVector2D const& startWorldPos, RealVector2D const& endWorldPos)
{
    if (Math::length(endWorldPos - startWorldPos) < NEAR_ZERO) {
        return;
    }
    _SimulationFacade::get()->cutConnections(startWorldPos, endWorldPos, _cutOnlyInSelection, EditorModel::get().isApplyToNetworks());
    EditorModel::get().update();
}
