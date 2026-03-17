#include "EditorController.h"

#include <memory>

#include <imgui.h>

#include <Base/Math.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/SimulationFacade.h>

#include "CreatorWindow.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "MainLoopEntityController.h"
#include "MultiplierWindow.h"
#include "OverlayController.h"
#include "PatternEditorWindow.h"
#include "SelectionWindow.h"
#include "StyleRepository.h"
#include "Viewport.h"

#include <GLFW/glfw3.h>

void EditorController::init()
{
    SelectionWindow::get().setup();
    EditorModel::get().setup();
    GenomeEditorWindow::get().setup();
    PatternEditorWindow::get().setup();
    CreatorWindow::get().setup();
    MultiplierWindow::get().setup();
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

    if (_SimulationFacade::get()->updateSelectionIfNecessary()) {
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
    Desc selectedData = _SimulationFacade::get()->getSelectedSimulationData(false);
    if (!onInspectObjects(DescEditService::get().getObjects(selectedData), false)) {
        showMessage(
            "Inspection not possible",
            "Too many objects are selected for inspection. A maximum of " + std::to_string(Const::MaxInspectedObjects) + " objects are allowed.");
    }
}

void EditorController::onInspectSelectedGenomes()
{
    Desc selectedData = _SimulationFacade::get()->getSelectedSimulationData(true);

    std::vector<GenomeDesc> uniqueGenomes;
    for (auto const& genome : selectedData._genomes) {
        auto genomeWithoutId = genome;
        genomeWithoutId._id = 0;
        bool alreadyCollected = false;
        for (auto const& existing : uniqueGenomes) {
            if (existing == genomeWithoutId) {
                alreadyCollected = true;
                break;
            }
        }
        if (!alreadyCollected) {
            uniqueGenomes.emplace_back(genomeWithoutId);
        }
    }

    for (auto const& genome : uniqueGenomes) {
        GenomeEditorWindow::get().openTab(genome);
    }

    printOverlayMessage(std::to_string(uniqueGenomes.size()) + " genome(s) inspected");
}

bool EditorController::onInspectObjects(std::vector<ExtendedObjectOrEnergyDesc> const& entities, bool selectGenomeTab)
{
    if (entities.empty()) {
        return true;
    }

    // Filter entities if cells are selected
    std::vector<ExtendedObjectOrEnergyDesc> filteredEntities;
    auto areCellsSelected = false;
    for (auto const& cellOrParticle : entities) {
        if (std::holds_alternative<ExtendedObjectDesc>(cellOrParticle)) {
            areCellsSelected = true;
            break;
        }
    }
    if (areCellsSelected) {
        for (auto const& cellOrParticle : entities) {
            if (std::holds_alternative<ExtendedObjectDesc>(cellOrParticle)) {
                filteredEntities.emplace_back(cellOrParticle);
            }
        }
    } else {
        filteredEntities = entities;
    }

    auto borderlessRendering = _SimulationFacade::get()->getSimulationParameters().borderlessRendering.value;

    std::set<uint64_t> inspectedIds;
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectedIds.insert(inspectorWindow->getId());
    }
    auto origInspectedIds = inspectedIds;
    for (auto const& entity : filteredEntities) {
        inspectedIds.insert(DescEditService::get().getId(entity));
    }

    std::vector<ExtendedObjectOrEnergyDesc> newEntities;
    for (auto const& entity : filteredEntities) {
        if (origInspectedIds.find(DescEditService::get().getId(entity)) == origInspectedIds.end()) {
            newEntities.emplace_back(entity);
        }
    }
    if (newEntities.empty()) {
        return true;
    }
    if (inspectedIds.size() > Const::MaxInspectedObjects) {
        return false;
    }
    RealVector2D center;
    int num = 0;
    for (auto const& entity : filteredEntities) {
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        center += entityPos;
        ++num;
    }
    center = center / num;

    float maxDistanceFromCenter = 0;
    for (auto const& entity : filteredEntities) {
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        auto distanceFromCenter = toFloat(Math::length(entityPos - center));
        maxDistanceFromCenter = std::max(maxDistanceFromCenter, distanceFromCenter);
    }
    auto viewSize = Viewport::get().getViewSize();
    auto factorX = maxDistanceFromCenter == 0 ? 1.0f : viewSize.x / maxDistanceFromCenter / 3.8f;
    auto factorY = maxDistanceFromCenter == 0 ? 1.0f : viewSize.y / maxDistanceFromCenter / 3.4f;

    for (auto const& entity : newEntities) {
        auto id = DescEditService::get().getId(entity);
        EditorModel::get().addInspectedEntity(entity);
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        auto windowPosX = (entityPos.x - center.x) * factorX + center.x;
        auto windowPosY = (entityPos.y - center.y) * factorY + center.y;
        windowPosX = std::min(std::max(windowPosX, 0.0f), toFloat(viewSize.x) - 300.0f) + 40.0f;
        windowPosY = std::min(std::max(windowPosY, 0.0f), toFloat(viewSize.y) - 500.0f) + 40.0f;
        _inspectorWindows.emplace_back(std::make_shared<_InspectorWindow>(id, RealVector2D{windowPosX, windowPosY}, selectGenomeTab));
    }
    return true;
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
    std::vector<ExtendedObjectOrEnergyDesc> inspectedEntities;
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (!inspectorWindow->isClosed()) {
            inspectorWindows.emplace_back(inspectorWindow);

            auto id = inspectorWindow->getId();
            inspectedEntities.emplace_back(EditorModel::get().getInspectedEntity(id));
        }
    }
    _inspectorWindows = inspectorWindows;
    EditorModel::get().setInspectedEntities(inspectedEntities);
    if (inspectedEntities.empty()) {
        return;
    }

    // Get inspected entities from simulation periodically
    static int counter = 0;
    if (++counter == 10) {
        std::vector<uint64_t> entityIds;
        for (auto const& entity : inspectedEntities) {
            entityIds.emplace_back(DescEditService::get().getId(entity));
        }
        auto inspectedData = _SimulationFacade::get()->getInspectedSimulationData(entityIds);
        auto newInspectedEntities = DescEditService::get().getObjects(inspectedData);
        EditorModel::get().setInspectedEntities(newInspectedEntities);
        counter = 0;
    }

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
        _SimulationFacade::get()->switchSelection(pos, std::max(0.5f, 10.0f / zoom));
    } else {
        _SimulationFacade::get()->swapSelection(pos, std::max(0.5f, 10.0f / zoom));
    }

    EditorModel::get().update();
}

void EditorController::onMoveSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::get().getZoomFactor();
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = EditorModel::get().isRolloutToClusters();
    updateData.posDeltaX = delta.x;
    updateData.posDeltaY = delta.y;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    EditorModel::get().update();
}

void EditorController::onFixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos, RealVector2D const& selectionPositionOnClick)
{
    auto shallowData = _SimulationFacade::get()->getSelectionShallowData();
    auto selectionPosition = RealVector2D{shallowData.centerPosX, shallowData.centerPosY};
    auto selectionDelta = selectionPosition - selectionPositionOnClick;

    auto mouseStart = Viewport::get().mapViewToWorldPosition(viewPos);
    auto mouseEnd = prevWorldPos;
    auto mouseDelta = mouseStart - mouseEnd;

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

void EditorController::onAccelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = true;
    updateData.velX = delta.x / 10;
    updateData.velY = delta.y / 10;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
}

void EditorController::onApplyForces(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::get().mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::get().getZoomFactor();
    _SimulationFacade::get()->applyForce_async(start, end, (end - start) / 200.0 * std::min(5.0f, zoom), 20.0f / zoom);
}

void EditorController::onUpdateSelectionRect(RealRect const& rect)
{
    auto startPos = Viewport::get().mapViewToWorldPosition(rect.topLeft);
    auto endPos = Viewport::get().mapViewToWorldPosition(rect.bottomRight);
    auto topLeft = RealVector2D{std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y)};
    auto bottomRight = RealVector2D{std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y)};

    _SimulationFacade::get()->setSelection(topLeft, bottomRight);
    EditorModel::get().update();
}
