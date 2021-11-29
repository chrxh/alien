#include "EditorController.h"

#include "imgui.h"

#include "EngineImpl/SimulationController.h"
#include "Viewport.h"
#include "StyleRepository.h"
#include "EditorModel.h"
#include "SelectionWindow.h"
#include "ActionsWindow.h"

_EditorController::_EditorController(
    SimulationController const& simController,
    Viewport const& viewport,
    StyleRepository const& styleRepository)
    : _simController(simController)
    , _viewport(viewport)
    , _styleRepository(styleRepository)
{
    _editorModel = boost::make_shared<_EditorModel>();
    _selectionWindow = boost::make_shared<_SelectionWindow>(_editorModel, _styleRepository);
    _actionsWindow = boost::make_shared<_ActionsWindow>(_editorModel, _styleRepository);
}

bool _EditorController::isOn() const
{
    return _on;
}

void _EditorController::setOn(bool value)
{
    _on = value;
}

void _EditorController::process()
{
    if (!_on) {
        return;
    }
    _selectionWindow->process();
    _actionsWindow->process();

    if (_selectionRect) {
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        auto startPos = _selectionRect->startPos;
        auto endPos = _selectionRect->endPos;
        draw_list->AddRectFilled(
            {startPos.x, startPos.y},
            {endPos.x, endPos.y},
            Const::SelectionAreaFillColor);
        draw_list->AddRect(
            {startPos.x, startPos.y},
            {endPos.x, endPos.y},
            Const::SelectionAreaBorderColor,
            0,
            0,
            1.0f);
    }

    if (!ImGui::GetIO().WantCaptureMouse) {
        auto mousePosImVec = ImGui::GetMousePos();
        RealVector2D mousePos{mousePosImVec.x, mousePosImVec.y};
        RealVector2D prevMousePosInt = _prevMousePosInt ? *_prevMousePosInt : mousePos;

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            leftMouseButtonPressed(mousePos);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            leftMouseButtonHold(mousePos, prevMousePosInt);
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            rightMouseButtonPressed(mousePos);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            rightMouseButtonHold(mousePos, prevMousePosInt);
        }

        _prevMousePosInt = mousePos;
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        leftMouseButtonReleased();
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        rightMouseButtonReleased();
    }

    synchronizeModelWithSimulation();
}

SelectionWindow _EditorController::getSelectionWindow() const
{
    return _selectionWindow;
}

ActionsWindow _EditorController::getActionsWindow() const
{
    return _actionsWindow;
}

void _EditorController::synchronizeModelWithSimulation()
{
    if (_simController->removeSelectionIfInvalid()) {
        _editorModel->clear();
    }
    {
        auto delta = _editorModel->getDeltaExtCenterPos();
        if (delta.x != 0 || delta.y != 0) {
            _simController->moveSelection(delta);

            auto selectionShallowData = _simController->getSelectionShallowData();
            _editorModel->setOrigSelectionShallowData(selectionShallowData);
        }
    }
    {
        auto delta = _editorModel->getDeltaExtCenterVel();
        if (delta.x != 0 || delta.y != 0) {
            _simController->accelerateSelection(delta);

            auto selectionShallowData = _simController->getSelectionShallowData();
            _editorModel->setOrigSelectionShallowData(selectionShallowData);
        }
    }
}

void _EditorController::leftMouseButtonPressed(RealVector2D const& viewPos)
{
    if (!_simController->isSimulationRunning()) {
        auto pos = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
        auto zoom = _viewport->getZoomFactor();
        _simController->switchSelection(pos, std::max(0.5f, 10.0f / zoom));

        auto selectionShallowData = _simController->getSelectionShallowData();
        _editorModel->setOrigSelectionShallowData(selectionShallowData);
    }
}

void _EditorController::leftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    auto start = _viewport->mapViewToWorldPosition({prevViewPos.x, prevViewPos.y});
    auto end = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = _viewport->getZoomFactor();
    if (_simController->isSimulationRunning()) {
        _simController->applyForce_async(start, end, (end - start) / 50.0 * std::min(5.0f, zoom), 20.0f / zoom);
    } else {
        auto delta = end - start;
        auto selectionData = _editorModel->getSelectionShallowData();
        selectionData.extCenterPosX += delta.x;
        selectionData.extCenterPosY += delta.y;
        _editorModel->setSelectionShallowData(selectionData);
    }
}

void _EditorController::leftMouseButtonReleased() {}

void _EditorController::rightMouseButtonPressed(RealVector2D const& viewPos)
{
    if (!_simController->isSimulationRunning()) {
        SelectionRect rect{viewPos, viewPos};
        _selectionRect = rect;
    }
}

void _EditorController::rightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (!_simController->isSimulationRunning()) {
        _selectionRect->endPos = viewPos;
        auto startPos = _viewport->mapViewToWorldPosition(_selectionRect->startPos);
        auto endPos = _viewport->mapViewToWorldPosition(_selectionRect->endPos);
        auto topLeft = RealVector2D{std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y)};
        auto bottomRight = RealVector2D{std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y)};

        _simController->setSelection(topLeft, bottomRight);
        auto selectionShallowData = _simController->getSelectionShallowData();
        _editorModel->setOrigSelectionShallowData(selectionShallowData);
    }
}

void _EditorController::rightMouseButtonReleased()
{
    if (!_simController->isSimulationRunning()) {
        _selectionRect = boost::none;
    }
}
