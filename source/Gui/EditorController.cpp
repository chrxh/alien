#include "EditorController.h"

#include <memory>
#include <imgui.h>

#include "Base/Math.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/DescriptionHelper.h"
#include "Viewport.h"
#include "StyleRepository.h"
#include "EditorModel.h"
#include "SelectionWindow.h"
#include "PatternEditorWindow.h"
#include "CreatorWindow.h"
#include "MultiplierWindow.h"
#include "GenomeEditorWindow.h"

_EditorController::_EditorController(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
{
    _editorModel = std::make_shared<_EditorModel>(_simController);
    _genomeEditorWindow = std::make_shared<_GenomeEditorWindow>(_editorModel, _simController);
    _selectionWindow = std::make_shared<_SelectionWindow>(_editorModel);
    _patternEditorWindow = std::make_shared<_PatternEditorWindow>(_editorModel, _simController, _viewport, this);
    _creatorWindow = std::make_shared<_CreatorWindow>(_editorModel, _simController, _viewport);
    _multiplierWindow = std::make_shared<_MultiplierWindow>(_editorModel, _simController, _viewport);
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
        _patternEditorWindow->process();
        _creatorWindow->process();
        _multiplierWindow->process();
        _genomeEditorWindow->process();
    }
    if (!_creatorWindow->isOn()) {
        _editorModel->setDrawMode(false);
    }
    
    processSelectionRect();
    processInspectorWindows();

    _editorModel->setForceNoRollout(ImGui::GetIO().KeyShift);

    processEvents();
}

SelectionWindow _EditorController::getSelectionWindow() const
{
    return _selectionWindow;
}

PatternEditorWindow _EditorController::getPatternEditorWindow() const
{
    return _patternEditorWindow;
}

CreatorWindow _EditorController::getCreatorWindow() const
{
    return _creatorWindow;
}

MultiplierWindow _EditorController::getMultiplierWindow() const
{
    return _multiplierWindow;
}

GenomeEditorWindow _EditorController::getGenomeEditorWindow() const
{
    return _genomeEditorWindow;
}

bool _EditorController::areInspectionWindowsActive() const
{
    return !_inspectorWindows.empty();
}

void _EditorController::onCloseAllInspectorWindows()
{
    _inspectorWindows.clear();
}

bool _EditorController::isObjectInspectionPossible() const
{
    return _patternEditorWindow->isObjectInspectionPossible();
}

bool _EditorController::isGenomeInspectionPossible() const
{
    return _patternEditorWindow->isGenomeInspectionPossible();
}

void _EditorController::onInspectSelectedObjects()
{
    DataDescription selectedData = _simController->getSelectedSimulationData(false);
    onInspectObjects(DescriptionHelper::getObjects(selectedData), false);
}

void _EditorController::onInspectSelectedGenomes()
{
    DataDescription selectedData = _simController->getSelectedSimulationData(true);
    auto constructors = DescriptionHelper::getConstructorToMainGenomes(selectedData);
    if (constructors.size() > 1) {
        constructors = {constructors.front()};
    }
    onInspectObjects(constructors, true);
}

void _EditorController::onInspectObjects(std::vector<CellOrParticleDescription> const& entities, bool selectGenomeTab)
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
    if (inspectedIds.size() > Const::MaxInspectedObjects) {
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
    auto factorX = maxDistanceFromCenter == 0 ? 1.0f : viewSize.x / maxDistanceFromCenter / 3.8f;
    auto factorY = maxDistanceFromCenter == 0 ? 1.0f : viewSize.y / maxDistanceFromCenter / 3.4f;

    for (auto const& entity : newEntities) {
        auto id = DescriptionHelper::getId(entity);
        _editorModel->addInspectedEntity(entity);
        auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
        auto windowPosX = (entityPos.x - center.x) * factorX + center.x;
        auto windowPosY = (entityPos.y - center.y) * factorY + center.y;
        windowPosX = std::min(std::max(windowPosX, 0.0f), toFloat(viewSize.x) - 300.0f) + 40.0f;
        windowPosY = std::min(std::max(windowPosY, 0.0f), toFloat(viewSize.y) - 500.0f) + 40.0f;
        _inspectorWindows.emplace_back(
            std::make_shared<_InspectorWindow>(_simController, _viewport, _editorModel, _genomeEditorWindow, id, RealVector2D{windowPosX, windowPosY}, selectGenomeTab));
    }
}

bool _EditorController::isCopyingPossible() const
{
    return _patternEditorWindow->isCopyingPossible();
}

void _EditorController::onCopy()
{
    _patternEditorWindow->onCopy();
}

bool _EditorController::isPastingPossible() const
{
    return _patternEditorWindow->isPastingPossible();
}

void _EditorController::onPaste()
{
    _patternEditorWindow->onPaste();
}

bool _EditorController::isDeletingPossible() const
{
    return _patternEditorWindow->isDeletingPossible();
}

void _EditorController::onDelete()
{
    _patternEditorWindow->onDelete();
}

void _EditorController::processEvents()
{
    auto running = _simController->isSimulationRunning();

    RealVector2D mousePos{ImGui::GetMousePos().x, ImGui::GetMousePos().y};
    RealVector2D prevMousePos = _prevMousePos ? *_prevMousePos : mousePos;

    if (!ImGui::GetIO().WantCaptureMouse) {

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (!running) {
                if (!_editorModel->isDrawMode()) {
                    selectObjects(mousePos, ImGui::GetIO().KeyCtrl);
                } else {
                    _creatorWindow->onDrawing();
                }
            } else {
                selectObjects(mousePos, ImGui::GetIO().KeyCtrl);
                _simController->setDetached(true);
                auto shallowData = _simController->getSelectionShallowData();
                _selectionPositionOnClick = {shallowData.centerPosX, shallowData.centerPosY};
                _mousePosOnClick = mousePos;
            }
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (!running) {
                if (!_editorModel->isDrawMode()) {
                    moveSelectedObjects(mousePos, prevMousePos);
                } else {
                    _creatorWindow->onDrawing();
                }
            } else {
                fixateSelectedObjects(mousePos, *_mousePosOnClick);
            }
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            if (!running && !_editorModel->isDrawMode()) {
                createSelectionRect(mousePos);
            }
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            if (!running && !_editorModel->isDrawMode()) {
                resizeSelectionRect(mousePos, prevMousePos);
            }
            if (running) {
                applyForces(mousePos, prevMousePos);
            }
        }
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        if (!running) {
            if (_editorModel->isDrawMode()) {
                _creatorWindow->finishDrawing();
            }
        } else {
            _simController->setDetached(false);
            accelerateSelectedObjects(mousePos, prevMousePos);
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        if (!running) {
            removeSelectionRect();
        }
    }

    if (_simController->updateSelectionIfNecessary()) {
        _editorModel->update();
    }
    _prevMousePos = mousePos;
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
    auto newInspectedEntities = DescriptionHelper::getObjects(inspectedData);
    _editorModel->setInspectedEntities(newInspectedEntities);

    inspectorWindows.clear();
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (_editorModel->existsInspectedEntity(inspectorWindow->getId())) {
            inspectorWindows.emplace_back(inspectorWindow);
        }
    }
    _inspectorWindows = inspectorWindows;
}

void _EditorController::selectObjects(RealVector2D const& viewPos, bool modifierKeyPressed)
{
    auto pos = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = _viewport->getZoomFactor();
    if (!modifierKeyPressed) {
        _simController->switchSelection(pos, std::max(0.5f, 10.0f / zoom));
    } else {
        _simController->swapSelection(pos, std::max(0.5f, 10.0f / zoom));
    }

    _editorModel->update();
}

void _EditorController::moveSelectedObjects(
    RealVector2D const& viewPos,
    RealVector2D const& prevViewPos)
{
    auto start = _viewport->mapViewToWorldPosition({prevViewPos.x, prevViewPos.y});
    auto end = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = _viewport->getZoomFactor();
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = _editorModel->isRolloutToClusters();
    updateData.posDeltaX = delta.x;
    updateData.posDeltaY = delta.y;
    _simController->shallowUpdateSelectedObjects(updateData);
    _editorModel->update();
}

void _EditorController::fixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    auto shallowData = _simController->getSelectionShallowData();
    auto selectionPosition = RealVector2D{shallowData.centerPosX, shallowData.centerPosY};
    auto selectionDelta = selectionPosition - *_selectionPositionOnClick;

    auto mouseStart = _viewport->mapViewToWorldPosition(viewPos);
    auto mouseEnd = _viewport->mapViewToWorldPosition(prevViewPos);
    auto mouseDelta = mouseStart - mouseEnd;

    auto selectionCorrectionDelta = mouseDelta - selectionDelta;
    auto worldSize = _simController->getWorldSize();
    if (Math::length(selectionCorrectionDelta) < std::min(worldSize.x, worldSize.y) / 2) {
        ShallowUpdateSelectionData updateData;
        updateData.considerClusters = true;
        updateData.posDeltaX = selectionCorrectionDelta.x;
        updateData.posDeltaY = selectionCorrectionDelta.y;
        _simController->shallowUpdateSelectedObjects(updateData);
    }
}

void _EditorController::accelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    auto start = _viewport->mapViewToWorldPosition({prevViewPos.x, prevViewPos.y});
    auto end = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto delta = end - start;

    auto zoom = _viewport->getZoomFactor();
    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = true;
    updateData.velDeltaX = delta.x / 10;
    updateData.velDeltaY = delta.y / 10;
    _simController->shallowUpdateSelectedObjects(updateData);
}

void _EditorController::applyForces(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    auto start = _viewport->mapViewToWorldPosition({prevViewPos.x, prevViewPos.y});
    auto end = _viewport->mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = _viewport->getZoomFactor();
    _simController->applyForce_async(start, end, (end - start) / 50.0 * std::min(5.0f, zoom), 20.0f / zoom);
}

void _EditorController::createSelectionRect(RealVector2D const& viewPos)
{
    SelectionRect rect{viewPos, viewPos};
    _selectionRect = rect;
}

void _EditorController::resizeSelectionRect(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    _selectionRect->endPos = viewPos;
    auto startPos = _viewport->mapViewToWorldPosition(_selectionRect->startPos);
    auto endPos = _viewport->mapViewToWorldPosition(_selectionRect->endPos);
    auto topLeft = RealVector2D{std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y)};
    auto bottomRight = RealVector2D{std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y)};

    _simController->setSelection(topLeft, bottomRight);
    _editorModel->update();
}

void _EditorController::removeSelectionRect()
{
    _selectionRect = std::nullopt;
}
