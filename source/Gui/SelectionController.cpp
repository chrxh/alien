#include "SelectionController.h"

#include <imgui.h>

#include <Base/Math.h>

#include <EngineInterface/SimulationFacade.h>

#include "EditorController.h"
#include "EditorModel.h"
#include "StyleService.h"
#include "Viewport.h"

void SelectionController::onLeftMouseButtonPressed(RealVector2D const& viewPos)
{
    selectObjects(viewPos, ImGui::GetIO().KeyCtrl);
    _worldPosOnClick = Viewport::get().mapViewToWorldPosition(viewPos);
    if (_SimulationFacade::get()->isSimulationRunning()) {
        _SimulationFacade::get()->setDetached(true);
    }

    auto shallowData = _SimulationFacade::get()->getSelectionShallowData();
    _selectionPositionOnClick = {shallowData.centerPosX, shallowData.centerPosY};
}

void SelectionController::onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    auto prevWorldPos = Viewport::get().mapViewToWorldPosition(prevViewPos);
    if (!_SimulationFacade::get()->isSimulationRunning()) {
        EditorController::get().onMoveSelectedObjectsBy(Viewport::get().mapViewToWorldPosition(viewPos) - prevWorldPos);
    } else if (_worldPosOnClick && _selectionPositionOnClick) {
        fixateSelectedObjects(viewPos, *_worldPosOnClick, *_selectionPositionOnClick);
    }
}

void SelectionController::onLeftMouseButtonReleased(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (_SimulationFacade::get()->isSimulationRunning()) {
        _SimulationFacade::get()->setDetached(false);
        accelerateSelectedObjects(viewPos, Viewport::get().mapViewToWorldPosition(prevViewPos));
    }
}

void SelectionController::onRightMouseButtonPressed(RealVector2D const& viewPos)
{
    _selectionRect = RealRect{viewPos, viewPos};
}

void SelectionController::onRightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (!ImGui::GetIO().KeyAlt && _selectionRect.has_value()) {
        _selectionRect->bottomRight = viewPos;
        updateSelectionRect(*_selectionRect);
    }
}

void SelectionController::onRightMouseButtonReleased()
{
    _selectionRect.reset();
}

void SelectionController::process()
{
    if (!_selectionRect) {
        return;
    }
    auto drawList = ImGui::GetBackgroundDrawList();
    auto startPos = _selectionRect->topLeft;
    auto endPos = _selectionRect->bottomRight;
    drawList->AddRectFilled({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaFillColor);
    drawList->AddRect({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaBorderColor, 0, 0, 1.0f);
}

void SelectionController::selectObjects(RealVector2D const& viewPos, bool modifierKeyPressed)
{
    auto pos = Viewport::get().mapViewToWorldPosition(viewPos);
    auto zoom = Viewport::get().getZoomFactor();
    if (!modifierKeyPressed) {
        _SimulationFacade::get()->switchSelection(pos, std::max(0.5f, 10.0f / zoom));
    } else {
        _SimulationFacade::get()->swapSelection(pos, std::max(0.5f, 10.0f / zoom));
    }
    EditorModel::get().update();
}

void SelectionController::fixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& worldPosOnClick, RealVector2D const& selectionPositionOnClick)
{
    auto shallowData = _SimulationFacade::get()->getSelectionShallowData();
    auto selectionPosition = RealVector2D{shallowData.centerPosX, shallowData.centerPosY};
    auto selectionDelta = selectionPosition - selectionPositionOnClick;

    auto mouseDelta = Viewport::get().mapViewToWorldPosition(viewPos) - worldPosOnClick;

    auto selectionCorrectionDelta = mouseDelta - selectionDelta;
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    if (Math::length(selectionCorrectionDelta) < std::min(worldSize.x, worldSize.y) / 2) {
        ShallowUpdateSelectionData updateData;
        updateData.considerClusters = true;
        updateData.posDeltaX = selectionCorrectionDelta.x;
        updateData.posDeltaY = selectionCorrectionDelta.y;
        _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    }
}

void SelectionController::accelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto delta = Viewport::get().mapViewToWorldPosition(viewPos) - prevWorldPos;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = true;
    updateData.velX = delta.x / 10;
    updateData.velY = delta.y / 10;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
}

void SelectionController::updateSelectionRect(RealRect const& rect)
{
    auto startPos = Viewport::get().mapViewToWorldPosition(rect.topLeft);
    auto endPos = Viewport::get().mapViewToWorldPosition(rect.bottomRight);
    auto topLeft = RealVector2D{std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y)};
    auto bottomRight = RealVector2D{std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y)};

    _SimulationFacade::get()->setSelection(topLeft, bottomRight);
    EditorModel::get().update();
}
