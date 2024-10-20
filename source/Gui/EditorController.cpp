#include "EditorController.h"

#include <memory>
#include <imgui.h>
#include <GLFW/glfw3.h>

#include "Base/Math.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/DescriptionEditService.h"
#include "Viewport.h"
#include "StyleRepository.h"
#include "EditorModel.h"
#include "SelectionWindow.h"
#include "PatternEditorWindow.h"
#include "CreatorWindow.h"
#include "MultiplierWindow.h"
#include "GenomeEditorWindow.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "MainLoopEntityController.h"

namespace
{
    auto const MaxInspectorWindowsToAdd = 10;
}

void EditorController::init(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    SelectionWindow::get().setup();
    EditorModel::get().setup(_simulationFacade);
    GenomeEditorWindow::get().setup(_simulationFacade);
    PatternEditorWindow::get().setup(_simulationFacade);
    CreatorWindow::get().setup(_simulationFacade);
    MultiplierWindow::get().setup(_simulationFacade);
}

bool EditorController::isOn() const
{
    return _on;
}

void EditorController::setOn(bool value)
{
    _on = value;
}

void EditorController::process()
{
    if (!_on) {
        return;
    }

    processInspectorWindows();

    EditorModel::get().setForceNoRollout(ImGui::GetIO().KeyShift);

    if (_simulationFacade->updateSelectionIfNecessary()) {
        EditorModel::get().update();
    }
}


bool EditorController::areInspectionWindowsActive() const
{
    return !_inspectorWindows.empty();
}

void EditorController::onCloseAllInspectorWindows()
{
    _inspectorWindows.clear();
}

void EditorController::onInspectSelectedObjects()
{
    auto selection = EditorModel::get().getSelectionShallowData();
    if (selection.numCells + selection.numParticles <= MaxInspectorWindowsToAdd) {
        DataDescription selectedData = _simulationFacade->getSelectedSimulationData(false);
        onInspectObjects(DescriptionEditService::get().getObjects(selectedData), false);
    } else {
        showMessage(
            "Inspection not possible",
            "Too many objects are selected for inspection. A maximum of " + std::to_string(MaxInspectorWindowsToAdd)
                + " objects are allowed.");
    }
}

void EditorController::onInspectSelectedGenomes()
{
    DataDescription selectedData = _simulationFacade->getSelectedSimulationData(true);
    auto constructors = DescriptionEditService::get().getConstructorToMainGenomes(selectedData);
    if (constructors.size() > 1) {
        constructors = {constructors.front()};
    }
    onInspectObjects(constructors, true);
}

void EditorController::onInspectObjects(std::vector<CellOrParticleDescription> const& entities, bool selectGenomeTab)
{
    if (entities.empty()) {
        return;
    }
    auto borderlessRendering = _simulationFacade->getSimulationParameters().borderlessRendering;

    std::set<uint64_t> inspectedIds;
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectedIds.insert(inspectorWindow->getId());
    }
    auto origInspectedIds = inspectedIds;
    for (auto const& entity : entities) {
        inspectedIds.insert(DescriptionEditService::get().getId(entity));
    }

    std::vector<CellOrParticleDescription> newEntities;
    for (auto const& entity : entities) {
        if (origInspectedIds.find(DescriptionEditService::get().getId(entity)) == origInspectedIds.end()) {
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
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescriptionEditService::get().getPos(entity), borderlessRendering);
        center += entityPos;
        ++num;
    }
    center = center / num;

    float maxDistanceFromCenter = 0;
    for (auto const& entity : entities) {
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescriptionEditService::get().getPos(entity), borderlessRendering);
        auto distanceFromCenter = toFloat(Math::length(entityPos - center));
        maxDistanceFromCenter = std::max(maxDistanceFromCenter, distanceFromCenter);
    }
    auto viewSize = Viewport::get().getViewSize();
    auto factorX = maxDistanceFromCenter == 0 ? 1.0f : viewSize.x / maxDistanceFromCenter / 3.8f;
    auto factorY = maxDistanceFromCenter == 0 ? 1.0f : viewSize.y / maxDistanceFromCenter / 3.4f;

    for (auto const& entity : newEntities) {
        auto id = DescriptionEditService::get().getId(entity);
        EditorModel::get().addInspectedEntity(entity);
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescriptionEditService::get().getPos(entity), borderlessRendering);
        auto windowPosX = (entityPos.x - center.x) * factorX + center.x;
        auto windowPosY = (entityPos.y - center.y) * factorY + center.y;
        windowPosX = std::min(std::max(windowPosX, 0.0f), toFloat(viewSize.x) - 300.0f) + 40.0f;
        windowPosY = std::min(std::max(windowPosY, 0.0f), toFloat(viewSize.y) - 500.0f) + 40.0f;
        _inspectorWindows.emplace_back(
            std::make_shared<_InspectorWindow>(_simulationFacade, id, RealVector2D{windowPosX, windowPosY}, selectGenomeTab));
    }
}

bool EditorController::isCopyingPossible() const
{
    return PatternEditorWindow::get().isCopyingPossible();
}

void EditorController::onCopy()
{
    PatternEditorWindow::get().onCopy();
    printOverlayMessage("Selection copied");
}

bool EditorController::isPastingPossible() const
{
    return PatternEditorWindow::get().isPastingPossible();
}

void EditorController::onPaste()
{
    PatternEditorWindow::get().onPaste();
    printOverlayMessage("Selection pasted");
}

bool EditorController::isDeletingPossible() const
{
    return PatternEditorWindow::get().isDeletingPossible();
}

void EditorController::onDelete()
{
    PatternEditorWindow::get().onDelete();
    printOverlayMessage("Selection deleted");
}

void EditorController::processInspectorWindows()
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
            inspectedEntities.emplace_back(EditorModel::get().getInspectedEntity(id));
        }
    }
    _inspectorWindows = inspectorWindows;
    EditorModel::get().setInspectedEntities(inspectedEntities);

    //update inspected entities from simulation
    if (inspectedEntities.empty()) {
        return;
    }
    std::vector<uint64_t> entityIds;
    for (auto const& entity : inspectedEntities) {
        entityIds.emplace_back(DescriptionEditService::get().getId(entity));
    }
    auto inspectedData = _simulationFacade->getInspectedSimulationData(entityIds);
    auto newInspectedEntities = DescriptionEditService::get().getObjects(inspectedData);
    EditorModel::get().setInspectedEntities(newInspectedEntities);

    inspectorWindows.clear();
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (EditorModel::get().existsInspectedEntity(inspectorWindow->getId())) {
            inspectorWindows.emplace_back(inspectorWindow);
        }
    }
    _inspectorWindows = inspectorWindows;
}

void EditorController::onSelectObjects(RealVector2D const& viewPos, bool modifierKeyPressed)
{
    auto pos = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::get().getZoomFactor();
    if (!modifierKeyPressed) {
        _simulationFacade->switchSelection(pos, std::max(0.5f, 10.0f / zoom));
    } else {
        _simulationFacade->swapSelection(pos, std::max(0.5f, 10.0f / zoom));
    }

    EditorModel::get().update();
}

void EditorController::onMoveSelectedObjects(
    RealVector2D const& viewPos,
    RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::get().getZoomFactor();
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = EditorModel::get().isRolloutToClusters();
    updateData.posDeltaX = delta.x;
    updateData.posDeltaY = delta.y;
    _simulationFacade->shallowUpdateSelectedObjects(updateData);
    EditorModel::get().update();
}

void EditorController::onFixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos, RealVector2D const& selectionPositionOnClick)
{
    auto shallowData = _simulationFacade->getSelectionShallowData(selectionPositionOnClick);
    auto selectionPosition = RealVector2D{shallowData.centerPosX, shallowData.centerPosY};
    auto selectionDelta = selectionPosition - selectionPositionOnClick;

    auto mouseStart = Viewport::get().mapViewToWorldPosition(viewPos);
    auto mouseEnd = prevWorldPos;
    auto mouseDelta = mouseStart - mouseEnd;

    auto selectionCorrectionDelta = mouseDelta - selectionDelta;
    auto worldSize = _simulationFacade->getWorldSize();
    if (Math::length(selectionCorrectionDelta) < std::min(worldSize.x, worldSize.y) / 2) {
        ShallowUpdateSelectionData updateData;
        updateData.considerClusters = true;
        updateData.posDeltaX = selectionCorrectionDelta.x;
        updateData.posDeltaY = selectionCorrectionDelta.y;
        _simulationFacade->shallowUpdateSelectedObjects(updateData);
    }
}

void EditorController::onAccelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = true;
    updateData.velDeltaX = delta.x / 10;
    updateData.velDeltaY = delta.y / 10;
    _simulationFacade->shallowUpdateSelectedObjects(updateData);
}

void EditorController::onApplyForces(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::get().getZoomFactor();
    _simulationFacade->applyForce_async(start, end, (end - start) / 50.0 * std::min(5.0f, zoom), 20.0f / zoom);
}

void EditorController::onUpdateSelectionRect(RealRect const& rect)
{
    auto startPos = Viewport::get().mapViewToWorldPosition(rect.topLeft);
    auto endPos = Viewport::get().mapViewToWorldPosition(rect.bottomRight);
    auto topLeft = RealVector2D{std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y)};
    auto bottomRight = RealVector2D{std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y)};

    _simulationFacade->setSelection(topLeft, bottomRight);
    EditorModel::get().update();
}
