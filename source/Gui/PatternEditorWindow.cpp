#include "PatternEditorWindow.h"

#include <ImFileDialog.h>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "EditorController.h"
#include "GenericFileDialogs.h"
#include "MessageDialog.h"
#include "Viewport.h"
#include "EngineInterface/SerializerService.h"

namespace
{
    auto const RightColumnWidth = 120.0f;
}

void PatternEditorWindow::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getString("editors.pattern editor.starting path", path.string());
}

void PatternEditorWindow::processIntern()
{

    auto selection = EditorModel::get().getSelectionShallowData();
    if (hasSelectionChanged(selection)) {
        _angle = 0;
        _angularVel = 0;
    }

    //load button
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
        onOpenPattern();
    }
    AlienImGui::Tooltip("Open pattern");

    //save button
    ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        onSavePattern();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Save pattern");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //copy button
    ImGui::SameLine();
    ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        onCopy();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Copy pattern");

    //paste button
    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedSelection.has_value());
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        onPaste();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste pattern");

    //delete button
    ImGui::SameLine();
    ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
    if (AlienImGui::ToolbarButton(ICON_FA_TRASH)) {
        onDelete();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete Pattern");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //inspect objects button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isObjectInspectionPossible());
    if (AlienImGui::ToolbarButton(ICON_FA_MICROSCOPE)) {
        EditorController::get().onInspectSelectedObjects();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Inspect Objects");

    //inspect genomes button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isGenomeInspectionPossible());
    if (AlienImGui::ToolbarButton(ICON_FA_DNA)) {
        EditorController::get().onInspectSelectedGenomes();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Inspect principal genome");

    if (ImGui::BeginChild(
        "##",
        ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)),
        false,
        ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
        AlienImGui::Group("Center position and velocity");

        auto const& selectionData = EditorModel::get().getSelectionShallowData();

        auto centerPosX = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position X")
                .textWidth(RightColumnWidth)
                .format("%.3f"),
            centerPosX);

        auto centerPosY = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position Y")
                .textWidth(RightColumnWidth)
                .format("%.3f"),
            centerPosY);

        auto centerVelX = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity X")
                .textWidth(RightColumnWidth)
                .step(0.1f)
                .format("%.3f"),
            centerVelX);

        auto centerVelY = EditorModel::get().isRolloutToClusters() ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity Y")
                .textWidth(RightColumnWidth)
                .step(0.1f)
                .format("%.3f"),
            centerVelY);

        AlienImGui::Group("Center rotation");
        auto origAngle = _angle;
        AlienImGui::SliderInputFloat(
            AlienImGui::SliderInputFloatParameters()
                .name("Angle")
                .textWidth(RightColumnWidth)
                .inputWidth(StyleRepository::get().scale(50.0f))
                .min(-180.0f)
                .max(180.0f)
                .format("%.1f"),
            _angle);

        auto origAngularVel = _angularVel;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
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
            updateData.velDeltaX = centerVelX - origCenterVelX;
            updateData.velDeltaY = centerVelY - origCenterVelY;
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
            updateData.angularVelDelta = _angularVel - origAngularVel;
            _simulationFacade->shallowUpdateSelectedObjects(updateData);
            EditorModel::get().update();
        }
        ImGui::EndDisabled();


        AlienImGui::Group("Color");
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
        AlienImGui::Group("Tools");
        ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());
        if (ImGui::Button(ICON_FA_WIND)) {
            _simulationFacade->uniformVelocitiesForSelectedObjects(EditorModel::get().isRolloutToClusters());
        }
        ImGui::EndDisabled();
        AlienImGui::Tooltip("Make uniform velocities");

        ImGui::SameLine();
        ImGui::BeginDisabled(EditorModel::get().isCellSelectionEmpty());
        if (ImGui::Button(ICON_FA_BALANCE_SCALE)) {
            _simulationFacade->relaxSelectedObjects(EditorModel::get().isRolloutToClusters());
        }
        AlienImGui::Tooltip("Release stresses");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_SORT_NUMERIC_DOWN)) {
            onGenerateExecutionOrderNumbers();
        }
        AlienImGui::Tooltip("Generate ascending execution order numbers");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TINT)) {
            onMakeSticky();
        }
        AlienImGui::Tooltip("Make sticky");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TINT_SLASH)) {
            onRemoveStickiness();
        }
        AlienImGui::Tooltip("Make unsticky");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_LINK)) {
            onSetBarrier(true);
        }
        AlienImGui::Tooltip("Convert to indestructible wall");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_UNLINK)) {
            onSetBarrier(false);
        }
        AlienImGui::Tooltip("Convert to destructible cell");
        ImGui::EndDisabled();

        _lastSelection = selection;
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    auto rolloutToClusters = EditorModel::get().isRolloutToClusters();
    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Roll out changes to cell networks"), rolloutToClusters)) {
        EditorModel::get().setRolloutToClusters(rolloutToClusters);
        _angle = 0;
        _angularVel = 0;
    }
    ImGui::SameLine();
    AlienImGui::HelpMarker("If turned on, all changes made in this window or with the mouse cursor are applied to the cell networks of the selected cell.\n"
                           "If this option is disabled, the changes will be applied only to the selected cells. In this case, the connections between the cells and the neighboring cells are recalculated when the positions are changed.\n"
                           "If you hold down the SHIFT key, this toggle button is temporarily turned off.");
}

void PatternEditorWindow::onOpenPattern()
{
    GenericFileDialogs::get().showOpenFileDialog(
        "Open pattern", "Pattern file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();
            ClusteredDataDescription content;
            if (SerializerService::deserializeContentFromFile(content, firstFilename.string())) {
                auto center = Viewport::get().getCenterInWorldPos();
                content.setCenter(center);
                _simulationFacade->addAndSelectSimulationData(DataDescription(content));
                EditorModel::get().update();
            } else {
                MessageDialog::get().information("Open pattern", "The selected file could not be opened.");
            }
        });
}

void PatternEditorWindow::onSavePattern()
{
    GenericFileDialogs::get().showSaveFileDialog(
        "Save pattern", "Pattern file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            auto content = _simulationFacade->getSelectedClusteredSimulationData(EditorModel::get().isRolloutToClusters());
            if (!SerializerService::serializeContentToFile(firstFilename.string(), content)) {
                MessageDialog::get().information("Save pattern", "The selected pattern could not be saved to the specified file.");
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
    DescriptionEditService::generateNewCreatureIds(data);
    _simulationFacade->addAndSelectSimulationData(data);
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
    GlobalSettings::get().setString("editors.pattern editor.starting path", _startingPath);
}

void PatternEditorWindow::onGenerateExecutionOrderNumbers()
{
    auto dataWithClusters = _simulationFacade->getSelectedSimulationData(true);
    auto dataWithoutClusters = _simulationFacade->getSelectedSimulationData(false);
    std::unordered_set<uint64_t> cellIds = dataWithoutClusters.getCellIds();

    auto parameters = _simulationFacade->getSimulationParameters();
    DescriptionEditService::generateExecutionOrderNumbers(dataWithClusters, cellIds, parameters.cellNumExecutionOrderNumbers);

    _simulationFacade->removeSelectedObjects(true);
    _simulationFacade->addAndSelectSimulationData(dataWithClusters);
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
    auto result = AlienImGui::ColorField(cellColor);
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
