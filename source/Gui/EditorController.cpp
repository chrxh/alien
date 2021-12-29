#include "EditorController.h"

#include <imgui.h>

#include "Base/Math.h"
#include "EngineImpl/SimulationController.h"
#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/DescriptionHelper.h"
#include "Viewport.h"
#include "StyleRepository.h"
#include "EditorModel.h"
#include "SelectionWindow.h"
#include "ManipulatorWindow.h"

_EditorController::_EditorController(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
{
    _editorModel = boost::make_shared<_EditorModel>(_simController);
    _selectionWindow = boost::make_shared<_SelectionWindow>(_editorModel);
    _manipulatorWindow = boost::make_shared<_ManipulatorWindow>(_editorModel, _simController, _viewport);
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

    if (!_simController->isSimulationRunning()) {
        _selectionWindow->process();
        _manipulatorWindow->process();
    }
    
    processSelectionRect();
    processInspectorWindows();

    if (!ImGui::GetIO().WantCaptureMouse) {
        auto mousePosImVec = ImGui::GetMousePos();
        RealVector2D mousePos{mousePosImVec.x, mousePosImVec.y};
        RealVector2D prevMousePosInt = _prevMousePosInt ? *_prevMousePosInt : mousePos;

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            leftMouseButtonPressed(mousePos, ImGui::GetIO().KeyCtrl);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            leftMouseButtonHold(mousePos, prevMousePosInt, ImGui::GetIO().KeyShift);
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

    if (_simController->updateSelectionIfNecessary()) {
        _editorModel->update();
    }
}

SelectionWindow _EditorController::getSelectionWindow() const
{
    return _selectionWindow;
}

ManipulatorWindow _EditorController::getManipulatorWindow() const
{
    return _manipulatorWindow;
}

void _EditorController::processSelectionRect()
{
    if (_selectionRect) {
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        auto startPos = _selectionRect->startPos;
        auto endPos = _selectionRect->endPos;
        draw_list->AddRectFilled({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaFillColor);
        draw_list->AddRect({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaBorderColor, 0, 0, 1.0f);
    }
}

void _EditorController::processInspectorWindows()
{
    //new entities to inspect
    newEntitiesToInspect(_editorModel->fetchEntitiesToInspect());

    //process inspector windows
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectorWindow->process();
    }

    //inspector windows closed?
    std::vector<InspectorWindow> inspectorWindows;
    std::vector<CellOrParticleDescription> inspectedEntities;
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (!inspectorWindow->isClosed()) {
            inspectorWindows.emplace_back(inspectorWindow);

            auto id = inspectorWindow->getId();
            inspectedEntities.emplace_back(_editorModel->getInspectedEntity(id));
        }
    }
    _inspectorWindows = inspectorWindows;
    _editorModel->setInspectedEntities(inspectedEntities);

    //update inspected entities from simulation
    if (inspectedEntities.empty()) {
        return;
    }
    std::vector<uint64_t> entityIds;
    for (auto const& entity : inspectedEntities) {
        entityIds.emplace_back(DescriptionHelper::getId(entity));
    }
    auto inspectedData = _simController->getInspectedSimulationData(entityIds);
    auto newInspectedEntities = DescriptionHelper::getEntities(inspectedData);
    _editorModel->setInspectedEntities(newInspectedEntities);

    inspectorWindows.clear();
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (_editorModel->existsInspectedEntity(inspectorWindow->getId())) {
            inspectorWindows.emplace_back(inspectorWindow);
        }
    }
    _inspectorWindows = inspectorWindows;
}

void _EditorController::newEntitiesToInspect(std::vector<CellOrParticleDescription> const& entities)
{
    if (entities.empty()) {
        return;
    }
    std::set<uint64_t> inspectedIds;
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectedIds.insert(inspectorWindow->getId());
    }
    auto origInspectedIds = inspectedIds;
    for (auto const& entity : entities) {
        inspectedIds.insert(DescriptionHelper::getId(entity));
    }

    std::vector<CellOrParticleDescription> newEntities;
    for (auto const& entity : entities) {
        if (origInspectedIds.find(DescriptionHelper::getId(entity)) == origInspectedIds.end()) {
            newEntities.emplace_back(entity);
        }
    }
    if (newEntities.empty()) {
        return;
    }
    if (inspectedIds.size() > Const::MaxInspectedEntities) {
        return;
    }
    RealVector2D center;
    int num = 0;
    for (auto const& entity : entities) {
        auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
        center += entityPos;
        ++num;
    }
    center = center / num;

    float maxDistanceFromCenter = 0;
    for (auto const& entity : entities) {
        auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
        auto distanceFromCenter = toFloat(Math::length(entityPos - center));
        maxDistanceFromCenter = std::max(maxDistanceFromCenter, distanceFromCenter);
    }
    auto viewSize = _viewport->getViewSize();
    auto viewRadius = std::min(viewSize.x, viewSize.y) / 2;
    auto factor = maxDistanceFromCenter == 0 ? 1.0f : viewRadius / maxDistanceFromCenter / 1.2f;

    for (auto const& entity : newEntities) {
        auto id = DescriptionHelper::getId(entity);
        _editorModel->addInspectedEntity(entity);
        auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
        auto windowPos = (entityPos - center) * factor + center;
        windowPos.x = std::min(std::max(windowPos.x, 0.0f), toFloat(viewSize.x) - 100.0f) + 40.0f;
        windowPos.y = std::min(std::max(windowPos.y, 0.0f), toFloat(viewSize.y) - 100.0f) + 40.0f;
        _inspectorWindows.emplace_back(
            boost::make_shared<_InspectorWindow>(_simController, _viewport, _editorModel, id, windowPos));
    }
}

void _EditorController::leftMouseButtonPressed(RealVector2D const& viewPos, bool modifierKeyPressed)
{
    if (!_simController->isSimulationRunning()) {
        auto pos = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
        auto zoom = _viewport->getZoomFactor();
        if (!modifierKeyPressed) {
            _simController->switchSelection(pos, std::max(0.5f, 10.0f / zoom));
        } else {
            _simController->swapSelection(pos, std::max(0.5f, 10.0f / zoom));
        }

        _editorModel->update();
    }
}

void _EditorController::leftMouseButtonHold(
    RealVector2D const& viewPos,
    RealVector2D const& prevViewPos,
    bool modifierKeyPressed)
{
    auto start = _viewport->mapViewToWorldPosition({prevViewPos.x, prevViewPos.y});
    auto end = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = _viewport->getZoomFactor();
    if (_simController->isSimulationRunning()) {
        _simController->applyForce_async(start, end, (end - start) / 50.0 * std::min(5.0f, zoom), 20.0f / zoom);
    } else {
        auto delta = end - start;

        ShallowUpdateSelectionData updateData;
        updateData.considerClusters = !modifierKeyPressed;
        updateData.posDeltaX = delta.x;
        updateData.posDeltaY = delta.y;
        _simController->shallowUpdateSelectedEntities(updateData);
        _editorModel->update();

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
        _editorModel->update();
    }
}

void _EditorController::rightMouseButtonReleased()
{
    if (!_simController->isSimulationRunning()) {
        _selectionRect = boost::none;
    }
}
