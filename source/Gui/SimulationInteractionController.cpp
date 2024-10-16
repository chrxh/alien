#include "SimulationInteractionController.h"

#include <imgui.h>
#include <cmath>

#include "Base/Resources.h"
#include "EngineInterface/SimulationFacade.h"

#include "OpenGLHelper.h"
#include "EditorController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "AlienImGui.h"
#include "EditorModel.h"
#include "SimulationView.h"
#include "CreatorWindow.h"

namespace
{
    auto constexpr CursorRadius = 13.0f;
}

_SimulationInteractionController::_SimulationInteractionController(
    SimulationFacade const& simulationFacade,
    EditorController const& editorController,
    SimulationView const& simulationView)
    : _simulationFacade(simulationFacade)
    , _editorController(editorController)
    , _simulationView(simulationView)
{
    _editorOn = OpenGLHelper::loadTexture(Const::EditorOnFilename);
    _editorOff = OpenGLHelper::loadTexture(Const::EditorOffFilename);
}

void _SimulationInteractionController::process()
{
    processEditWidget();

    if (_modes.editMode) {
        processSelectionRect();
    }
    if (!_editorController->getCreatorWindow()->isOn()) {
        _modes.drawMode = false;
    }
    processEvents();
}

bool _SimulationInteractionController::isEditMode() const
{
    return _modes.editMode;
}

void _SimulationInteractionController::setEditMode(bool value)
{
    _modes.editMode = value;
    _editorController->setOn(_modes.editMode);
}

bool _SimulationInteractionController::isDrawMode() const
{
    return _modes.drawMode;
}

void _SimulationInteractionController::setDrawMode(bool value)
{
    _modes.drawMode = value;
}

bool _SimulationInteractionController::isPositionSelectionMode() const
{
    return _modes.positionSelectionMode;
}

void _SimulationInteractionController::setPositionSelectionMode(bool value)
{
    _modes.positionSelectionMode = value;
}

std::optional<RealVector2D> _SimulationInteractionController::getPositionSelectionData() const
{
    if (ImGui::GetIO().WantCaptureMouse) {
        return std::nullopt;
    }

    auto mousePos = ImGui::GetMousePos();
    return Viewport::get().mapViewToWorldPosition({mousePos.x, mousePos.y});
}

void _SimulationInteractionController::processEditWidget()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + viewport->Size.y - scale(120.0f)));
    ImGui::SetNextWindowSize(ImVec2(scale(160.0f), scale(100.0f)));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("TOOLBAR", NULL, windowFlags);

    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor());
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor());
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor());

    auto actionTexture = _modes.editMode ? _editorOn.textureId : _editorOff.textureId;
    if (ImGui::ImageButton((void*)(intptr_t)actionTexture, {scale(80.0f), scale(80.0f)}, {0, 0}, {1.0f, 1.0f})) {
        _modes.editMode = !_modes.editMode;
        _editorController->setOn(!_editorController->isOn());
    }

    ImGui::PopStyleColor(3);
    ImGui::End();
}

void _SimulationInteractionController::processEvents()
{
    auto mousePos = ImGui::GetMousePos();
    IntVector2D mousePosInt{toInt(mousePos.x), toInt(mousePos.y)};
    IntVector2D prevMousePosInt = _prevMousePosInt ? *_prevMousePosInt : mousePosInt;

    if (!ImGui::GetIO().WantCaptureMouse) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            leftMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            leftMouseButtonHold(mousePosInt, prevMousePosInt);
        }
        if (ImGui::GetIO().MouseWheel > 0) {
            mouseWheelUp(mousePosInt, std::abs(ImGui::GetIO().MouseWheel));
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            leftMouseButtonReleased(mousePosInt, prevMousePosInt);
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            rightMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            rightMouseButtonHold(mousePosInt, prevMousePosInt);
        }
        if (ImGui::GetIO().MouseWheel < 0) {
            mouseWheelDown(mousePosInt, std::abs(ImGui::GetIO().MouseWheel));
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            rightMouseButtonReleased();
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            middleMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            middleMouseButtonHold(mousePosInt);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
            middleMouseButtonReleased();
        }
        drawCursor();
    }
    processMouseWheel(mousePosInt);

    _prevMousePosInt = mousePosInt;
}

void _SimulationInteractionController::leftMouseButtonPressed(IntVector2D const& mousePos)
{
    _modesAtClick = _modes;

    if (_modes.positionSelectionMode) {
        _modes.positionSelectionMode = false;
        return;
    }

    if (!_modes.editMode) {
        _lastZoomTimepoint.reset();
        _simulationView->setMotionBlur(_simulationView->getMotionBlur() * 2);
    } else {
        if (!ImGui::GetIO().KeyAlt) {
            if (!_modes.drawMode) {
                _editorController->onSelectObjects(toRealVector2D(mousePos), ImGui::GetIO().KeyCtrl);
                _worldPosOnClick = Viewport::get().mapViewToWorldPosition(toRealVector2D(mousePos));
                if (_simulationFacade->isSimulationRunning()) {
                    _simulationFacade->setDetached(true);
                }

                auto shallowData = _simulationFacade->getSelectionShallowData(*_worldPosOnClick);
                _selectionPositionOnClick = {shallowData.centerPosX, shallowData.centerPosY};
            } else {
                _editorController->getCreatorWindow()->onDrawing();
            }
        }
    }
}

void _SimulationInteractionController::leftMouseButtonHold(IntVector2D const& mousePos, IntVector2D const& prevMousePos)
{
    if (_modesAtClick.positionSelectionMode) {
        return;
    }

    if (!_modesAtClick.editMode) {
        Viewport::get().zoom(mousePos, calcZoomFactor(_lastZoomTimepoint ? *_lastZoomTimepoint : std::chrono::steady_clock::now()));
    } else {
        RealVector2D prevWorldPos = Viewport::get().mapViewToWorldPosition(toRealVector2D(prevMousePos));

        if (!_modesAtClick.drawMode) {
            if (!_simulationFacade->isSimulationRunning()) {
                _editorController->onMoveSelectedObjects(toRealVector2D(mousePos), prevWorldPos);
            } else {
                _editorController->onFixateSelectedObjects(toRealVector2D(mousePos), *_worldPosOnClick, *_selectionPositionOnClick);
            }
        } else {
            _editorController->getCreatorWindow()->onDrawing();
        }
    }
}

void _SimulationInteractionController::mouseWheelUp(IntVector2D const& mousePos, float strongness)
{
    _mouseWheelAction =
        MouseWheelAction{.up = true, .strongness = strongness, .start = std::chrono::steady_clock::now(), .lastTime = std::chrono::steady_clock::now()};
}

void _SimulationInteractionController::leftMouseButtonReleased(IntVector2D const& mousePos, IntVector2D const& prevMousePos)
{
    if (_modesAtClick.positionSelectionMode) {
        return;
    }

    if (!_modesAtClick.editMode) {
        _simulationView->setMotionBlur(_simulationView->getMotionBlur() / 2);
    } else {
        if (_modesAtClick.drawMode) {
            _editorController->getCreatorWindow()->finishDrawing();
        } else {
            if (_simulationFacade->isSimulationRunning()) {
                _simulationFacade->setDetached(false);
                RealVector2D prevWorldPos = Viewport::get().mapViewToWorldPosition(toRealVector2D(prevMousePos));
                _editorController->onAccelerateSelectedObjects(toRealVector2D(mousePos), prevWorldPos);
            }
        }
    }
}

void _SimulationInteractionController::rightMouseButtonPressed(IntVector2D const& mousePos)
{
    _modesAtClick = _modes;

    if (_modes.positionSelectionMode) {
        _modes.positionSelectionMode = false;
        return;
    }

    if (!_modes.editMode) {
        _lastZoomTimepoint.reset();
        _simulationView->setMotionBlur(_simulationView->getMotionBlur() * 2);
    } else {
        if (!ImGui::GetIO().KeyAlt) {
            if (!_simulationFacade->isSimulationRunning() && !_modes.drawMode) {
                auto viewPos = toRealVector2D(mousePos);
                RealRect rect{viewPos, viewPos};
                _selectionRect = rect;
            }
        }
    }
}

void _SimulationInteractionController::rightMouseButtonHold(IntVector2D const& mousePos, IntVector2D const& prevMousePos)
{
    if (_modesAtClick.positionSelectionMode) {
        return;
    }

    if (!_modesAtClick.editMode) {
        Viewport::get().zoom(mousePos, 1.0f / calcZoomFactor(_lastZoomTimepoint ? *_lastZoomTimepoint : std::chrono::steady_clock::now()));
    } else {
        if (!ImGui::GetIO().KeyAlt) {
            auto isSimulationRunning = _simulationFacade->isSimulationRunning();
            if (!isSimulationRunning && !_modesAtClick.drawMode && _selectionRect.has_value()) {
                _selectionRect->bottomRight = toRealVector2D(mousePos);
                _editorController->onUpdateSelectionRect(*_selectionRect);
            }
            if (isSimulationRunning) {
                RealVector2D prevWorldPos = Viewport::get().mapViewToWorldPosition(toRealVector2D(prevMousePos));
                _editorController->onApplyForces(toRealVector2D(mousePos), prevWorldPos);
            }
        }
    }
}

void _SimulationInteractionController::mouseWheelDown(IntVector2D const& mousePos, float strongness)
{
    _mouseWheelAction =
        MouseWheelAction{.up = false, .strongness = strongness, .start = std::chrono::steady_clock::now(), .lastTime = std::chrono::steady_clock::now()};
}

void _SimulationInteractionController::rightMouseButtonReleased()
{
    if (_modesAtClick.positionSelectionMode) {
        return;
    }

    if (!_modesAtClick.editMode) {
        _simulationView->setMotionBlur(_simulationView->getMotionBlur() / 2);
    } else {
        if (!_simulationFacade->isSimulationRunning()) {
            _selectionRect.reset();
        }
    }
}

void _SimulationInteractionController::processMouseWheel(IntVector2D const& mousePos)
{
    if (_mouseWheelAction) {
        auto zoomFactor = powf(calcZoomFactor(_mouseWheelAction->lastTime), 2.2f * _mouseWheelAction->strongness);
        auto now = std::chrono::steady_clock::now();
        _mouseWheelAction->lastTime = now;
        Viewport::get().zoom(mousePos, _mouseWheelAction->up ? zoomFactor : 1.0f / zoomFactor);
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - _mouseWheelAction->start).count() > 100) {
            _mouseWheelAction.reset();
        }
    }
}

void _SimulationInteractionController::middleMouseButtonPressed(IntVector2D const& mousePos)
{
    _worldPosForMovement = Viewport::get().mapViewToWorldPosition({toFloat(mousePos.x), toFloat(mousePos.y)});
}

void _SimulationInteractionController::middleMouseButtonHold(IntVector2D const& mousePos)
{
    Viewport::get().centerTo(*_worldPosForMovement, mousePos);
}

void _SimulationInteractionController::middleMouseButtonReleased()
{
    _worldPosForMovement = std::nullopt;
}

void _SimulationInteractionController::drawCursor()
{
    auto mousePos = ImGui::GetMousePos();
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto editorModel = _editorController->getEditorModel();

    if (!ImGui::GetIO().WantCaptureMouse) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    }

    // position selection cursor
    if (_modes.positionSelectionMode) {
        auto cursorSize = scale(CursorRadius);

        // shadow
        drawList->AddRectFilled(
            {mousePos.x - scale(2.0f), mousePos.y - cursorSize}, {mousePos.x + scale(2.0f), mousePos.y - cursorSize / 2}, Const::CursorShadowColor);
        drawList->AddRectFilled(
            {mousePos.x - scale(2.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(2.0f), mousePos.y + cursorSize}, Const::CursorShadowColor);
        drawList->AddRectFilled(
            {mousePos.x - cursorSize, mousePos.y - scale(2.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);
        drawList->AddRectFilled(
            {mousePos.x + cursorSize / 2, mousePos.y - scale(2.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);

        // foreground
        drawList->AddRectFilled(
            {mousePos.x - scale(1.0f), mousePos.y - cursorSize}, {mousePos.x + scale(1.0f), mousePos.y - cursorSize / 2}, Const::CursorColor);
        drawList->AddRectFilled(
            {mousePos.x - scale(1.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(1.0f), mousePos.y + cursorSize}, Const::CursorColor);
        drawList->AddRectFilled(
            {mousePos.x - cursorSize, mousePos.y - scale(1.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(1.0f)}, Const::CursorColor);
        drawList->AddRectFilled(
            {mousePos.x + cursorSize / 2, mousePos.y - scale(1.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(1.0f)}, Const::CursorColor);
        return;
    }

    // editing cursors
    if (_modes.editMode) {
        if (!_modes.drawMode || _simulationFacade->isSimulationRunning()) {
            auto cursorSize = scale(CursorRadius);

            // shadow
            drawList->AddRectFilled(
                {mousePos.x - scale(2.0f), mousePos.y - cursorSize}, {mousePos.x + scale(2.0f), mousePos.y - cursorSize / 2}, Const::CursorShadowColor);
            drawList->AddRectFilled(
                {mousePos.x - scale(2.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(2.0f), mousePos.y + cursorSize}, Const::CursorShadowColor);
            drawList->AddRectFilled(
                {mousePos.x - cursorSize, mousePos.y - scale(2.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);
            drawList->AddRectFilled(
                {mousePos.x + cursorSize / 2, mousePos.y - scale(2.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);

            // foreground
            drawList->AddRectFilled(
                {mousePos.x - scale(1.0f), mousePos.y - cursorSize}, {mousePos.x + scale(1.0f), mousePos.y - cursorSize / 2}, Const::CursorColor);
            drawList->AddRectFilled(
                {mousePos.x - scale(1.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(1.0f), mousePos.y + cursorSize}, Const::CursorColor);
            drawList->AddRectFilled(
                {mousePos.x - cursorSize, mousePos.y - scale(1.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(1.0f)}, Const::CursorColor);
            drawList->AddRectFilled(
                {mousePos.x + cursorSize / 2, mousePos.y - scale(1.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(1.0f)}, Const::CursorColor);
        } else {
            auto zoom = Viewport::get().getZoomFactor();
            auto radius = editorModel->getPencilWidth() * zoom;
            auto color = Const::IndividualCellColors[editorModel->getDefaultColorCode()];
            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(color, h, s, v);
            drawList->AddCircleFilled(mousePos, radius, ImColor::HSV(h, s, v, 0.6f));
        }
        return;
    }

    // navigation cursor
    if (!_modes.editMode) {
        auto cursorSize = scale(CursorRadius);

        // shadow
        drawList->AddCircle(mousePos, cursorSize / 2, Const::CursorShadowColor, 0, scale(4.0f));
        drawList->AddLine(
            {mousePos.x + sqrtf(2.0f) / 2.0f * cursorSize / 2, mousePos.y + sqrtf(2.0f) / 2.0f * cursorSize / 2},
            {mousePos.x + cursorSize, mousePos.y + cursorSize},
            Const::CursorShadowColor,
            scale(4.0f));

        // foreground
        drawList->AddCircle(mousePos, cursorSize / 2, Const::CursorColor, 0, scale(2.0f));
        drawList->AddLine(
            {mousePos.x + sqrtf(2.0f) / 2.0f * cursorSize / 2, mousePos.y + sqrtf(2.0f) / 2.0f * cursorSize / 2},
            {mousePos.x + cursorSize, mousePos.y + cursorSize},
            Const::CursorColor,
            scale(2.0f));
    }
}

void _SimulationInteractionController::processSelectionRect()
{
    if (_selectionRect) {
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        auto startPos = _selectionRect->topLeft;
        auto endPos = _selectionRect->bottomRight;
        draw_list->AddRectFilled({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaFillColor);
        draw_list->AddRect({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaBorderColor, 0, 0, 1.0f);
    }
}

float _SimulationInteractionController::calcZoomFactor(std::chrono::steady_clock::time_point const& lastTimepoint)
{
    auto now = std::chrono::steady_clock::now();
    auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTimepoint).count());
    _lastZoomTimepoint = now;
    return powf(Viewport::get().getZoomSensitivity(), duration / 15);
}
