#include "EditorController.h"

#include "imgui.h"

#include "EngineImpl/SimulationController.h"
#include "Viewport.h"

_EditorController::_EditorController(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
{
    _selectionRect = SelectionRect{{100.0f, 100.0f}, {200.0f, 300.0f}};
}

bool _EditorController::isOn() const
{
    return _on;;
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

    if (!ImGui::GetIO().WantCaptureMouse) {
        auto mousePos = ImGui::GetMousePos();
        IntVector2D mousePosInt{toInt(mousePos.x), toInt(mousePos.y)};
        IntVector2D prevMousePosInt = _prevMousePosInt ? *_prevMousePosInt : mousePosInt;

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            leftMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            leftMouseButtonHold(mousePosInt, prevMousePosInt);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            leftMouseButtonReleased();
        }

        _prevMousePosInt = mousePosInt;
    }
}

void _EditorController::leftMouseButtonPressed(IntVector2D const& viewPos)
{
    if (!_simController->isSimulationRunning()) {
        auto pos = _viewport->mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
        auto zoom = _viewport->getZoomFactor();
        _simController->switchSelection(pos, 10.0f / zoom);
    }
}

void _EditorController::leftMouseButtonHold(IntVector2D const& viewPos, IntVector2D const& prevViewPos)
{
    auto start = _viewport->mapViewToWorldPosition({toFloat(prevViewPos.x), toFloat(prevViewPos.y)});
    auto end = _viewport->mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
    auto zoom = _viewport->getZoomFactor();
    if (_simController->isSimulationRunning()) {
        _simController->applyForce_async(start, end, (end - start) / 30, 20.0f / zoom);
    } else {
        _simController->moveSelection(end - start);
    }
}

void _EditorController::leftMouseButtonReleased() {}
