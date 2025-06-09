#include "PatternEditorWindow.h"

#include <ImFileDialog.h>
#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "AlienGui.h"
#include "EditorController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "Viewport.h"
#include "PersisterInterface/SerializerService.h"

namespace
{
    auto const RightColumnWidth = 120.0f;
}

void PatternEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getValue("editors.pattern editor.starting path", path.string());
}

void PatternEditorWindow::processIntern()
{

    auto selection = EditorModel::get().getSelectionShallowData();
    if (hasSelectionChanged(selection)) {
        _angle = 0;
        _angularVel = 0;
    }

    //load button
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN))) {
        onOpenPattern();
    }
    AlienGui::Tooltip("Open pattern");

    //save button
    ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_SAVE))) {
        onSavePattern();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip("Save pattern");

    ImGui::SameLine();
    AlienGui::ToolbarSeparator();

    //copy button
    ImGui::SameLine();
    ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_COPY))) {
        onCopy();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip("Copy pattern");

    //paste button
    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedSelection.has_value());
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_PASTE))) {
        onPaste();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip("Paste pattern");

    //delete button
    ImGui::SameLine();
    ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_TRASH))) {
        onDelete();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip("Delete Pattern");

    ImGui::SameLine();
    AlienGui::ToolbarSeparator();

    //inspect objects button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isObjectInspectionPossible());
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_MICROSCOPE))) {
        EditorController::get().onInspectSelectedObjects();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip("Inspect Objects");

    //inspect genomes button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isGenomeInspectionPossible());
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_DNA))) {
        EditorController::get().onInspectSelectedGenomes();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip("Inspect principal genome");

    if (ImGui::BeginChild(
        "##",
        ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)),
        false,
        ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
        AlienGui::Group("Center position and velocity");

        auto const& selectionData = EditorModel::get().getSelectionShallowData();

        auto centerPosX = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Position X")
                .textWidth(RightColumnWidth)
                .format("%.3f"),
            centerPosX);

        auto centerPosY = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Position Y")
                .textWidth(RightColumnWidth)
                .format("%.3f"),
            centerPosY);

        auto centerVelX = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Velocity X")
                .textWidth(RightColumnWidth)
                .step(0.1f)
                .format("%.3f"),
            centerVelX);

        auto centerVelY = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Velocity Y")
                .textWidth(RightColumnWidth)
                .step(0.1f)
                .format("%.3f"),
            centerVelY);

        AlienGui::Group("Center rotation");
        auto origAngle = _angle;
        AlienGui::SliderInputFloat(
            AlienGui::SliderInputFloatParameters()
                .name("Angle")
                .textWidth(RightColumnWidth)
                .inputWidth(StyleRepository::get().scale(50.0f))
                .min(-180.0f)
                .max(180.0f)
                .format("%.1f"),
            _angle);

        auto origAngularVel = _angularVel;
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Angular velocity")
                .textWidth(RightColumnWidth)
                .step(0.01f)
                .format("%.2f"),
            _angularVel);

        if (centerPosX != origCenterPosX || centerPosY != origCenterPosY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = EditorModel::get().isRolloutToClusters();
            updateData.posDeltaX = centerPosX - origCenterPosX;
            updateData.posDeltaY = centerPosY - origCenterPosY;
            _simulationFacade->shallowUpdateSelectedObjects(updateData);
            EditorModel::get().update();
        }

        if (centerVelX != origCenterVelX || centerVelY != origCenterVelY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = EditorModel::get().isRolloutToClusters();
            updateData.velX = centerVelX;
            updateData.velY = centerVelY;
            _simulationFacade->shallowUpdateSelectedObjects(updateData);
            EditorModel::get().update();
        }

        if (_angle != origAngle) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = EditorModel::get().isRolloutToClusters();
            updateData.angleDelta = _angle - origAngle;
            _simulationFacade->shallowUpdateSelectedObjects(updateData);
            EditorModel::get().update();
        }

        if (_angularVel != origAngularVel) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = EditorModel::get().isRolloutToClusters();
            updateData.angularVel = _angularVel;
            _simulationFacade->shallowUpdateSelectedObjects(updateData);
            EditorModel::get().update();
        }
        ImGui::EndDisabled();


        AlienGui::Group("Color");
        if (colorButton("    ##color1", Const::IndividualCellColor1)) {
            _simulationFacade->colorSelectedObjects(0, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(0);
        }
        ImGui::SameLine();
        if (colorButton("    ##color2", Const::IndividualCellColor2)) {
            _simulationFacade->colorSelectedObjects(1, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(1);
        }
        ImGui::SameLine();
        if (colorButton("    ##color3", Const::IndividualCellColor3)) {
            _simulationFacade->colorSelectedObjects(2, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(2);
        }
        ImGui::SameLine();
        if (colorButton("    ##color4", Const::IndividualCellColor4)) {
            _simulationFacade->colorSelectedObjects(3, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(3);
        }
        ImGui::SameLine();
        if (colorButton("    ##color5", Const::IndividualCellColor5)) {
            _simulationFacade->colorSelectedObjects(4, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(4);
        }
        ImGui::SameLine();
        if (colorButton("    ##color6", Const::IndividualCellColor6)) {
            _simulationFacade->colorSelectedObjects(5, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(5);
        }
        ImGui::SameLine();
        if (colorButton("    ##color7", Const::IndividualCellColor7)) {
            _simulationFacade->colorSelectedObjects(6, EditorModel::get().isRolloutToClusters());
            EditorModel::get().setDefaultColorCode(6);
        }
        AlienGui::Group("Tools");
        ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
        if (ImGui::Button(ICON_FA_WIND)) {
            _simulationFacade->uniformVelocitiesForSelectedObjects(EditorModel::get().isRolloutToClusters());
        }
        ImGui::EndDisabled();
        AlienGui::Tooltip("Make uniform velocities");

        ImGui::SameLine();
        ImGui::BeginDisabled(EditorModel::get().isCellSelectionEmpty());
        if (ImGui::Button(ICON_FA_BALANCE_SCALE)) {
            _simulationFacade->relaxSelectedObjects(EditorModel::get().isRolloutToClusters());
        }
        AlienGui::Tooltip("Release stresses");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TINT)) {
            onMakeSticky();
        }
        AlienGui::Tooltip("Make sticky");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TINT_SLASH)) {
            onRemoveStickiness();
        }
        AlienGui::Tooltip("Make unsticky");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_LINK)) {
            onSetBarrier(true);
        }
        AlienGui::Tooltip("Convert to indestructible wall");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_UNLINK)) {
            onSetBarrier(false);
        }
        AlienGui::Tooltip("Convert to destructible cell");
        ImGui::EndDisabled();

        _lastSelection = selection;
    }
    ImGui::EndChild();

    AlienGui::Separator();
    auto rolloutToClusters = EditorModel::get().isRolloutToClusters();
    if (AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Roll out changes to cell networks"), rolloutToClusters)) {
        EditorModel::get().setRolloutToClusters(rolloutToClusters);
        _angle = 0;
        _angularVel = 0;
    }
    ImGui::SameLine();
    AlienGui::HelpMarker("If turned on, all changes made in this window or with the mouse cursor are applied to the cell networks of the selected cell.\n"
                           "If this option is disabled, the changes will be applied only to the selected cells. In this case, the connections between the cells and the neighboring cells are recalculated when the positions are changed.\n"
                           "If you hold down the SHIFT key, this toggle button is temporarily turned off.");
}

bool PatternEditorWindow::isShown()
{
    return _on && EditorController::get().isOn();
}

void PatternEditorWindow::onOpenPattern()
{
    GenericFileDialog::get().showOpenFileDialog(
        "Open pattern", "Pattern file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();
            CollectionDescription content;
            if (SerializerService::get().deserializeContentFromFile(content, firstFilename.string())) {
                auto center = Viewport::get().getCenterInWorldPos();
                content.setCenter(center);
                _simulationFacade->addAndSelectSimulationData(CollectionDescription(content));
                EditorModel::get().update();
            } else {
                GenericMessageDialog::get().information("Open pattern", "The selected file could not be opened.");
            }
        });
}

void PatternEditorWindow::onSavePattern()
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save pattern", "Pattern file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            auto content = _simulationFacade->getSelectedSimulationData(EditorModel::get().isRolloutToClusters());
            if (!SerializerService::get().serializeContentToFile(firstFilename.string(), content)) {
                GenericMessageDialog::get().information("Save pattern", "The selected pattern could not be saved to the specified file.");
            }
        });
}

bool PatternEditorWindow::isObjectInspectionPossible() const
{
    return !EditorModel::get().isSelectionEmpty();
}

bool PatternEditorWindow::isGenomeInspectionPossible() const
{
    return !EditorModel::get().isSelectionEmpty();
}

bool PatternEditorWindow::isCopyingPossible() const
{
    return !EditorModel::get().isSelectionEmpty();
}

void PatternEditorWindow::onCopy()
{
    _copiedSelection = _simulationFacade->getSelectedSimulationData(EditorModel::get().isRolloutToClusters());
}

bool PatternEditorWindow::isPastingPossible() const
{
    return _copiedSelection.has_value();
}

void PatternEditorWindow::onPaste()
{
    auto data = *_copiedSelection;
    auto center = Viewport::get().getCenterInWorldPos();
    data.setCenter(center);
    _simulationFacade->addAndSelectSimulationData(std::move(data));
    EditorModel::get().update();
}

bool PatternEditorWindow::isDeletingPossible() const
{
    return !EditorModel::get().isSelectionEmpty() && !EditorModel::get().areEntitiesInspected();
}

void PatternEditorWindow::onDelete()
{
    _simulationFacade->removeSelectedObjects(EditorModel::get().isRolloutToClusters());
    EditorModel::get().update();
}

PatternEditorWindow::PatternEditorWindow()
    : AlienWindow("Pattern editor", "editors.pattern editor", true)
{}

void PatternEditorWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("editors.pattern editor.starting path", _startingPath);
}

void PatternEditorWindow::onMakeSticky()
{
    _simulationFacade->makeSticky(EditorModel::get().isRolloutToClusters());
}

void PatternEditorWindow::onRemoveStickiness()
{
    _simulationFacade->removeStickiness(EditorModel::get().isRolloutToClusters());
}

void PatternEditorWindow::onSetBarrier(bool value)
{
    _simulationFacade->setBarrier(value, EditorModel::get().isRolloutToClusters());
}

bool PatternEditorWindow::colorButton(std::string id, uint32_t cellColor)
{
    ImGui::PushID(id.c_str());
    auto result = AlienGui::ColorField(cellColor);
    ImGui::PopID();
    return result;
}

bool PatternEditorWindow::hasSelectionChanged(SelectionShallowData const& selection) const
{
    if(!_lastSelection) {
        return false;
    }
    return _lastSelection->numCells != selection.numCells || _lastSelection->numParticles != selection.numParticles;
}
