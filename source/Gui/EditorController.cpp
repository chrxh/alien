#include "EditorController.h"

#include <chrono>

#include <imgui.h>

#include <Base/GlobalSettings.h>

#include <Data/DescEditService.h>

#include <EngineInterface/SimulationFacade.h>

#include "CreatorController.h"
#include "EditorModel.h"
#include "EditorWidget.h"
#include "ForceController.h"
#include "GenomeEditorWindow.h"
#include "InspectionController.h"
#include "MultiplierWidget.h"
#include "OverlayController.h"
#include "ScissorsController.h"
#include "SelectionController.h"
#include "SelectionWidget.h"
#include "Viewport.h"

namespace
{
    auto constexpr SelectionRolloutInterval = std::chrono::milliseconds(500);
}

void EditorController::init()
{
    EditorModel::get().setup();
    GenomeEditorWindow::get().setup();
    InspectionController::get().setup();
    SelectionController::get().setup();
    ScissorsController::get().setup();
    ForceController::get().setup();
    CreatorController::get().setup();
    MultiplierWidget::get().setup();
    SelectionWidget::get().setup();
    EditorWidget::get().setup();

    auto& settings = GlobalSettings::get();
    auto& model = EditorModel::get();
    model.setTool(settings.getValue("editors.tool", model.getTool()));
    model.setApplyToNetworks(settings.getValue("editors.apply to networks", model.isApplyToNetworksPersistent()));
    model.setGlueOnContact(settings.getValue("editors.glue on contact", model.isGlueOnContact()));
}

void EditorController::shutdown()
{
    auto& settings = GlobalSettings::get();
    auto const& model = EditorModel::get();
    settings.setValue("editors.tool", model.getTool());
    settings.setValue("editors.apply to networks", model.isApplyToNetworksPersistent());
    settings.setValue("editors.glue on contact", model.isGlueOnContact());
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

    auto const& io = ImGui::GetIO();
    EditorModel::get().setScopeInvertedTemporarily(io.KeyShift && !io.WantTextInput);

    auto const& simulationFacade = _SimulationFacade::get();
    auto& model = EditorModel::get();
    if (simulationFacade->updateSelectionIfNecessary()) {
        model.update();
    } else if (simulationFacade->isSimulationRunning() && !model.isSelectionEmpty()) {
        auto now = std::chrono::steady_clock::now();
        if (now - _lastSelectionRolloutTime >= SelectionRolloutInterval) {
            simulationFacade->updateSelection();
            _lastSelectionRolloutTime = now;
        }
        model.update();
    }
}

bool EditorController::isCopyingPossible() const
{
    return !EditorModel::get().isSelectionEmpty();
}

void EditorController::onCopy()
{
    _copiedSelection = _SimulationFacade::get()->getSelectedSimulationData(EditorModel::get().isApplyToNetworks());
    printOverlayMessage("Selection copied");
}

bool EditorController::isPastingPossible() const
{
    return _copiedSelection.has_value();
}

void EditorController::onPaste()
{
    auto content = *_copiedSelection;
    DescEditService::get().setCenter(content, Viewport::get().getCenterInWorldPos());
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    EditorModel::get().update();
    printOverlayMessage("Selection pasted");
}

bool EditorController::isDeletingPossible() const
{
    return !EditorModel::get().isSelectionEmpty() && !EditorModel::get().areEntitiesInspected();
}

void EditorController::onDelete()
{
    _SimulationFacade::get()->removeSelectedObjects(EditorModel::get().isApplyToNetworks());
    EditorModel::get().update();
    printOverlayMessage("Selection deleted");
}

bool EditorController::isDeselectingPossible() const
{
    return !EditorModel::get().isSelectionEmpty() && !CreatorController::get().hasPoints();
}

void EditorController::onDeselect()
{
    _SimulationFacade::get()->removeSelection();
    EditorModel::get().update();
}

void EditorController::onColorSelectedObjects(int color)
{
    _SimulationFacade::get()->colorSelectedObjects(toUInt8(color), EditorModel::get().isApplyToNetworks());
    EditorModel::get().setDefaultColorCode(color);
}

void EditorController::onSetSticky(bool value)
{
    if (value) {
        _SimulationFacade::get()->makeSticky(EditorModel::get().isApplyToNetworks());
    } else {
        _SimulationFacade::get()->removeStickiness(EditorModel::get().isApplyToNetworks());
    }
}

void EditorController::onSetStatic(bool value)
{
    _SimulationFacade::get()->setStatic(value, EditorModel::get().isApplyToNetworks());
    if (value) {
        onUniformVelocities();
    }
}

void EditorController::onUniformVelocities()
{
    _SimulationFacade::get()->uniformVelocitiesForSelectedObjects(EditorModel::get().isApplyToNetworks());
    EditorModel::get().update();
}

void EditorController::onReleaseStresses()
{
    _SimulationFacade::get()->relaxSelectedObjects(EditorModel::get().isApplyToNetworks());
}

void EditorController::onGlueSelectedObjects()
{
    _SimulationFacade::get()->glueSelectedObjects(EditorModel::get().isApplyToNetworks());
    EditorModel::get().update();
    printOverlayMessage("Selection glued");
}

void EditorController::onMoveSelectedObjectsBy(RealVector2D const& delta)
{
    auto const& model = EditorModel::get();
    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = model.isApplyToNetworks();
    updateData.glueOnContact = model.isGlueOnContact();
    updateData.posDeltaX = delta.x;
    updateData.posDeltaY = delta.y;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    EditorModel::get().update();
}

void EditorController::onRotateSelectedObjects(float angleDelta)
{
    auto const& model = EditorModel::get();
    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = model.isApplyToNetworks();
    updateData.glueOnContact = model.isGlueOnContact();
    updateData.angleDelta = angleDelta;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    EditorModel::get().update();
}

void EditorController::onSetVelocityOfSelectedObjects(RealVector2D const& velocity)
{
    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = EditorModel::get().isApplyToNetworks();
    updateData.velX = velocity.x;
    updateData.velY = velocity.y;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    EditorModel::get().update();
}

void EditorController::onSetAngularVelocityOfSelectedObjects(float angularVelocity)
{
    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = EditorModel::get().isApplyToNetworks();
    updateData.angularVel = angularVelocity;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    EditorModel::get().update();
}
