#include "ForceController.h"

#include <EngineInterface/SimulationFacade.h>

#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr ForceRadius = 20.0f;
}

void ForceController::onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (ImGui::GetIO().KeyAlt || !_SimulationFacade::get()->isSimulationRunning()) {
        return;
    }
    auto start = Viewport::get().mapViewToWorldPosition(prevViewPos);
    auto end = Viewport::get().mapViewToWorldPosition(viewPos);
    auto zoom = Viewport::get().getZoomFactor();
    _SimulationFacade::get()->applyForce_async(start, end, (end - start) / 200.0 * std::min(5.0f, zoom), ForceRadius / zoom);
}

void ForceController::drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const
{
    drawList->AddCircle(mousePos, ForceRadius, Const::ConstructionPreviewHintLineColor, 0, scale(1.5f));
}
