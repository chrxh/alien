#include "PatternEditorWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"
#include "Viewport.h"
#include "SavePatternDialog.h"
#include "OpenPatternDialog.h"

namespace
{
    auto const MaxInspectorWindowsToAdd = 10;
    auto const MaxContentTextWidth = 120.0f;
}

_PatternEditorWindow::_PatternEditorWindow(
    EditorModel const& editorModel,
    SimulationController const& simController,
    Viewport const& viewport)
    : _AlienWindow("Pattern editor", "editor.pattern editor", true)
    , _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{
    _savePatternDialog = std::make_shared<_SavePatternDialog>(simController);
    _openPatternDialog = std::make_shared<_OpenPatternDialog>(editorModel, simController, viewport);
}

void _PatternEditorWindow::processIntern()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});

    auto selection = _editorModel->getSelectionShallowData();
    if (hasSelectionChanged(selection)) {
        _angle = 0;
        _angularVel = 0;
    }

    //load button
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
        _openPatternDialog->show();
    }

    //save button
    ImGui::BeginDisabled(!isCopyingPossible());
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        _savePatternDialog->show(_editorModel->isRolloutToClusters());
    }
    ImGui::EndDisabled();

    //copy button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isCopyingPossible());
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        onCopy();
    }
    ImGui::EndDisabled();

    //paste button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isPastingPossible());
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        onPaste();
    }
    ImGui::EndDisabled();

    //delete button
    ImGui::SameLine();
    ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
    if (AlienImGui::ToolbarButton(ICON_FA_TRASH)) {
        _simController->removeSelectedEntities(_editorModel->isRolloutToClusters());
        _editorModel->update();
    }
    ImGui::EndDisabled();

    //inspector button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isInspectionPossible());
    if (AlienImGui::ToolbarButton(ICON_FA_MICROSCOPE)) {
        onInspectEntities();
    }
    ImGui::EndDisabled();

    if (ImGui::BeginChild(
        "##",
        ImVec2(0, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(50.0f)),
        false,
        ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
        AlienImGui::Group("Center position and velocity");

        auto const& selectionData = _editorModel->getSelectionShallowData();

        auto centerPosX = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position X")
                .textWidth(MaxContentTextWidth)
                .format("%.2f"),
            centerPosX);

        auto centerPosY = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position Y")
                .textWidth(MaxContentTextWidth)
                .format("%.2f"),
            centerPosY);

        auto centerVelX = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity X")
                .textWidth(MaxContentTextWidth)
                .step(0.1f)
                .format("%.2f"),
            centerVelX);

        auto centerVelY = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity Y")
                .textWidth(MaxContentTextWidth)
                .step(0.1f)
                .format("%.2f"),
            centerVelY);

        AlienImGui::Group("Center rotation");
        auto origAngle = _angle;
        AlienImGui::SliderInputFloat(
            AlienImGui::SliderInputFloatParameters()
                .name("Angle")
                .textWidth(MaxContentTextWidth)
                .inputWidth(StyleRepository::getInstance().scaleContent(50))
                .min(-180.0f)
                .max(180.0f)
                .format("%.1f"),
            _angle);

        auto origAngularVel = _angularVel;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Angular velocity")
                .textWidth(MaxContentTextWidth)
                .step(0.01f)
                .format("%.2f"),
            _angularVel);

        if (centerPosX != origCenterPosX || centerPosY != origCenterPosY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.posDeltaX = centerPosX - origCenterPosX;
            updateData.posDeltaY = centerPosY - origCenterPosY;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        if (centerVelX != origCenterVelX || centerVelY != origCenterVelY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.velDeltaX = centerVelX - origCenterVelX;
            updateData.velDeltaY = centerVelY - origCenterVelY;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        if (_angle != origAngle) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.angleDelta = _angle - origAngle;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        if (_angularVel != origAngularVel) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.angularVelDelta = _angularVel - origAngularVel;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        AlienImGui::Group("Color");
        if (colorButton("    ##color1", Const::IndividualCellColor1)) {
            _simController->colorSelectedEntities(0, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(0);
        }
        ImGui::SameLine();
        if (colorButton("    ##color2", Const::IndividualCellColor2)) {
            _simController->colorSelectedEntities(1, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(1);
        }
        ImGui::SameLine();
        if (colorButton("    ##color3", Const::IndividualCellColor3)) {
            _simController->colorSelectedEntities(2, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(2);
        }
        ImGui::SameLine();
        if (colorButton("    ##color4", Const::IndividualCellColor4)) {
            _simController->colorSelectedEntities(3, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(3);
        }
        ImGui::SameLine();
        if (colorButton("    ##color5", Const::IndividualCellColor5)) {
            _simController->colorSelectedEntities(4, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(4);
        }
        ImGui::SameLine();
        if (colorButton("    ##color6", Const::IndividualCellColor6)) {
            _simController->colorSelectedEntities(5, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(5);
        }
        ImGui::SameLine();
        if (colorButton("    ##color7", Const::IndividualCellColor7)) {
            _simController->colorSelectedEntities(6, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(7);
        }
        AlienImGui::Group("Further actions");
        if (ImGui::Button("Set uniform velocities")) {
            _simController->uniformVelocitiesForSelectedEntities(_editorModel->isRolloutToClusters());
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(_editorModel->isCellSelectionEmpty());

        if (ImGui::Button("Release tensions")) {
            _simController->relaxSelectedEntities(_editorModel->isRolloutToClusters());
        }
        if (ImGui::Button("Generate ascending branch numbers")) {
            onGenerateBranchNumbers();
        }
        ImGui::EndDisabled();

        _lastSelection = selection;
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();

    AlienImGui::Separator();
    auto rolloutToClusters = _editorModel->isRolloutToClusters();
    if (AlienImGui::ToggleButton("Roll out to cell clusters", rolloutToClusters)) {
        _editorModel->setRolloutToClusters(rolloutToClusters);
        _angle = 0;
        _angularVel = 0;
    }
    ImGui::SameLine();
    AlienImGui::HelpMarker("If turned on, all changes made in this window or with the mouse cursor are applied to the cell clusters of the selected cell.\n"
                           "If this option is disabled, the changes will be applied only to the selected cells. In this case, the connections between the cells and the neighboring cells are recalculated when the positions are changed.\n"
                           "If you hold down the SHIFT key, this toggle button is temporarily turned off.");

    _savePatternDialog->process();
    _openPatternDialog->process();
}

bool _PatternEditorWindow::isInspectionPossible() const
{
    auto selection = _editorModel->getSelectionShallowData();
    return !_editorModel->isSelectionEmpty() && selection.numCells + selection.numParticles <= MaxInspectorWindowsToAdd;
}

void _PatternEditorWindow::onInspectEntities()
{
    DataDescription selectedData = _simController->getSelectedSimulationData(false);
    _editorModel->inspectEntities(DescriptionHelper::getEntities(selectedData));
}

bool _PatternEditorWindow::isCopyingPossible() const
{
    return !_editorModel->isSelectionEmpty() && !_editorModel->areEntitiesInspected();
}

void _PatternEditorWindow::onCopy()
{
    _copiedSelection = _simController->getSelectedSimulationData(_editorModel->isRolloutToClusters());
}

bool _PatternEditorWindow::isPastingPossible() const
{
    return _copiedSelection.has_value() && !_editorModel->areEntitiesInspected();
}

void _PatternEditorWindow::onPaste()
{
    auto data = *_copiedSelection;
    auto center = _viewport->getCenterInWorldPos();
    data.setCenter(center);
    _simController->addAndSelectSimulationData(data);
    _editorModel->update();
}

void _PatternEditorWindow::onGenerateBranchNumbers()
{
    auto dataWithClusters = _simController->getSelectedSimulationData(true);
    auto dataWithoutClusters = _simController->getSelectedSimulationData(false);
    std::unordered_set<uint64_t> cellIds = dataWithoutClusters.getCellIds();

    auto parameters = _simController->getSimulationParameters();
    DescriptionHelper::generateBranchNumbers(dataWithClusters, cellIds, parameters.cellMaxTokenBranchNumber);

    _simController->removeSelectedEntities(true);
    _simController->addAndSelectSimulationData(dataWithClusters);
}

bool _PatternEditorWindow::colorButton(std::string id, uint32_t cellColor)
{
    float h, s, v;
    AlienImGui::convertRGBtoHSV(cellColor, h, s,v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.6f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.7f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 1.0f));
    auto result = ImGui::Button(id.c_str());
    ImGui::PopStyleColor(3);

    return result;
}

bool _PatternEditorWindow::hasSelectionChanged(SelectionShallowData const& selection) const
{
    if(!_lastSelection) {
        return false;
    }
    return _lastSelection->numCells != selection.numCells || _lastSelection->numParticles != selection.numParticles;
}
