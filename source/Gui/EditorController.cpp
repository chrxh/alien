#include "EditorController.h"

#include <memory>
#include <imgui.h>
#include <GLFW/glfw3.h>

#include "Base/Math.h"
#include "EngineInterface/SimulationController.h"
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
#include "MessageDialog.h"
#include "OverlayMessageController.h"

namespace
{
    auto const MaxInspectorWindowsToAdd = 10;
}

_EditorController::_EditorController(SimulationController const& simController)
    : _simController(simController)
{
    _editorModel = std::make_shared<_EditorModel>(_simController);
    _genomeEditorWindow = std::make_shared<_GenomeEditorWindow>(_editorModel, _simController);
    _selectionWindow = std::make_shared<_SelectionWindow>(_editorModel);
    _patternEditorWindow = std::make_shared<_PatternEditorWindow>(_editorModel, _simController, this);
    _creatorWindow = std::make_shared<_CreatorWindow>(_editorModel, _simController);
    _multiplierWindow = std::make_shared<_MultiplierWindow>(_editorModel, _simController);
}

void _EditorController::registerCyclicReferences(
    UploadSimulationDialogWeakPtr const& uploadSimulationDialog,
    SimulationInteractionControllerWeakPtr const& simulationInteractionController)
{
    _genomeEditorWindow->registerCyclicReferences(uploadSimulationDialog);
    _creatorWindow->registerCyclicReferences(simulationInteractionController);
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
    _patternEditorWindow->process();
    _creatorWindow->process();
    _multiplierWindow->process();
    _genomeEditorWindow->process();

    processInspectorWindows();

    _editorModel->setForceNoRollout(ImGui::GetIO().KeyShift);

    if (_simController->updateSelectionIfNecessary()) {
        _editorModel->update();
    }
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

EditorModel _EditorController::getEditorModel() const
{
    return _editorModel;
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
    auto selection = _editorModel->getSelectionShallowData();
    if (selection.numCells + selection.numParticles <= MaxInspectorWindowsToAdd) {
        DataDescription selectedData = _simController->getSelectedSimulationData(false);
        onInspectObjects(DescriptionEditService::getObjects(selectedData), false);
    } else {
        showMessage(
            "Inspection not possible",
            "Too many objects are selected for inspection. A maximum of " + std::to_string(MaxInspectorWindowsToAdd)
                + " objects are allowed.");
    }
}

void _EditorController::onInspectSelectedGenomes()
{
    DataDescription selectedData = _simController->getSelectedSimulationData(true);
    auto constructors = DescriptionEditService::getConstructorToMainGenomes(selectedData);
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
    auto borderlessRendering = _simController->getSimulationParameters().borderlessRendering;

    std::set<uint64_t> inspectedIds;
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectedIds.insert(inspectorWindow->getId());
    }
    auto origInspectedIds = inspectedIds;
    for (auto const& entity : entities) {
        inspectedIds.insert(DescriptionEditService::getId(entity));
    }

    std::vector<CellOrParticleDescription> newEntities;
    for (auto const& entity : entities) {
        if (origInspectedIds.find(DescriptionEditService::getId(entity)) == origInspectedIds.end()) {
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
        auto entityPos = Viewport::mapWorldToViewPosition(DescriptionEditService::getPos(entity), borderlessRendering);
        center += entityPos;
        ++num;
    }
    center = center / num;

    float maxDistanceFromCenter = 0;
    for (auto const& entity : entities) {
        auto entityPos = Viewport::mapWorldToViewPosition(DescriptionEditService::getPos(entity), borderlessRendering);
        auto distanceFromCenter = toFloat(Math::length(entityPos - center));
        maxDistanceFromCenter = std::max(maxDistanceFromCenter, distanceFromCenter);
    }
    auto viewSize = Viewport::getViewSize();
    auto factorX = maxDistanceFromCenter == 0 ? 1.0f : viewSize.x / maxDistanceFromCenter / 3.8f;
    auto factorY = maxDistanceFromCenter == 0 ? 1.0f : viewSize.y / maxDistanceFromCenter / 3.4f;

    for (auto const& entity : newEntities) {
        auto id = DescriptionEditService::getId(entity);
        _editorModel->addInspectedEntity(entity);
        auto entityPos = Viewport::mapWorldToViewPosition(DescriptionEditService::getPos(entity), borderlessRendering);
        auto windowPosX = (entityPos.x - center.x) * factorX + center.x;
        auto windowPosY = (entityPos.y - center.y) * factorY + center.y;
        windowPosX = std::min(std::max(windowPosX, 0.0f), toFloat(viewSize.x) - 300.0f) + 40.0f;
        windowPosY = std::min(std::max(windowPosY, 0.0f), toFloat(viewSize.y) - 500.0f) + 40.0f;
        _inspectorWindows.emplace_back(
            std::make_shared<_InspectorWindow>(_simController, _editorModel, _genomeEditorWindow, id, RealVector2D{windowPosX, windowPosY}, selectGenomeTab));
    }
}

bool _EditorController::isCopyingPossible() const
{
    return _patternEditorWindow->isCopyingPossible();
}

void _EditorController::onCopy()
{
    _patternEditorWindow->onCopy();
    printOverlayMessage("Selection copied");
}

bool _EditorController::isPastingPossible() const
{
    return _patternEditorWindow->isPastingPossible();
}

void _EditorController::onPaste()
{
    _patternEditorWindow->onPaste();
    printOverlayMessage("Selection pasted");
}

bool _EditorController::isDeletingPossible() const
{
    return _patternEditorWindow->isDeletingPossible();
}

void _EditorController::onDelete()
{
    _patternEditorWindow->onDelete();
    printOverlayMessage("Selection deleted");
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
        entityIds.emplace_back(DescriptionEditService::getId(entity));
    }
    auto inspectedData = _simController->getInspectedSimulationData(entityIds);
    auto newInspectedEntities = DescriptionEditService::getObjects(inspectedData);
    _editorModel->setInspectedEntities(newInspectedEntities);

    inspectorWindows.clear();
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (_editorModel->existsInspectedEntity(inspectorWindow->getId())) {
            inspectorWindows.emplace_back(inspectorWindow);
        }
    }
    _inspectorWindows = inspectorWindows;
}

void _EditorController::onSelectObjects(RealVector2D const& viewPos, bool modifierKeyPressed)
{
    auto pos = Viewport::mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::getZoomFactor();
    if (!modifierKeyPressed) {
        _simController->switchSelection(pos, std::max(0.5f, 10.0f / zoom));
    } else {
        _simController->swapSelection(pos, std::max(0.5f, 10.0f / zoom));
    }

    _editorModel->update();
}

void _EditorController::onMoveSelectedObjects(
    RealVector2D const& viewPos,
    RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::getZoomFactor();
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = _editorModel->isRolloutToClusters();
    updateData.posDeltaX = delta.x;
    updateData.posDeltaY = delta.y;
    _simController->shallowUpdateSelectedObjects(updateData);
    _editorModel->update();
}

void _EditorController::onFixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos, RealVector2D const& selectionPositionOnClick)
{
    auto shallowData = _simController->getSelectionShallowData(selectionPositionOnClick);
    auto selectionPosition = RealVector2D{shallowData.centerPosX, shallowData.centerPosY};
    auto selectionDelta = selectionPosition - selectionPositionOnClick;

    auto mouseStart = Viewport::mapViewToWorldPosition(viewPos);
    auto mouseEnd = prevWorldPos;
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

void _EditorController::onAccelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto delta = end - start;

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = true;
    updateData.velDeltaX = delta.x / 10;
    updateData.velDeltaY = delta.y / 10;
    _simController->shallowUpdateSelectedObjects(updateData);
}

void _EditorController::onApplyForces(RealVector2D const& viewPos, RealVector2D const& prevWorldPos)
{
    auto start = prevWorldPos;
    auto end = Viewport::mapViewToWorldPosition({viewPos.x, viewPos.y});
    auto zoom = Viewport::getZoomFactor();
    _simController->applyForce_async(start, end, (end - start) / 50.0 * std::min(5.0f, zoom), 20.0f / zoom);
}

void _EditorController::onUpdateSelectionRect(RealRect const& rect)
{
    auto startPos = Viewport::mapViewToWorldPosition(rect.topLeft);
    auto endPos = Viewport::mapViewToWorldPosition(rect.bottomRight);
    auto topLeft = RealVector2D{std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y)};
    auto bottomRight = RealVector2D{std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y)};

    _simController->setSelection(topLeft, bottomRight);
    _editorModel->update();
}
