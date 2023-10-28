#include "PatternEditorWindow.h"

#include <ImFileDialog.h>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationController.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "EditorController.h"
#include "GenericFileDialogs.h"
#include "MessageDialog.h"
#include "Viewport.h"
#include "EngineInterface/Serializer.h"

namespace
{
    auto const RightColumnWidth = 120.0f;
}

_PatternEditorWindow::_PatternEditorWindow(
    EditorModel const& editorModel,
    SimulationController const& simController,
    Viewport const& viewport,
    EditorControllerWeakPtr const& editorController)
    : _AlienWindow("Pattern editor", "editors.pattern editor", true)
    , _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
    , _editorController(editorController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("editors.pattern editor.starting path", path.string());
}

_PatternEditorWindow::~_PatternEditorWindow()
{
    GlobalSettings::getInstance().setStringState("editors.pattern editor.starting path", _startingPath);
}

void _PatternEditorWindow::processIntern()
{

    auto selection = _editorModel->getSelectionShallowData();
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
    ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
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
    ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
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
    ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
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
        _editorController->onInspectSelectedObjects();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Inspect Objects");

    //inspect genomes button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isGenomeInspectionPossible());
    if (AlienImGui::ToolbarButton(ICON_FA_DNA)) {
        _editorController->onInspectSelectedGenomes();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Inspect principal genome");

    if (ImGui::BeginChild(
        "##",
        ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)),
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
                .textWidth(RightColumnWidth)
                .format("%.3f"),
            centerPosX);

        auto centerPosY = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position Y")
                .textWidth(RightColumnWidth)
                .format("%.3f"),
            centerPosY);

        auto centerVelX = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity X")
                .textWidth(RightColumnWidth)
                .step(0.1f)
                .format("%.3f"),
            centerVelX);

        auto centerVelY = _editorModel->isRolloutToClusters() ? selectionData.clusterCenterVelY : selectionData.centerVelY;
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
                .inputWidth(StyleRepository::getInstance().scale(50))
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
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.posDeltaX = centerPosX - origCenterPosX;
            updateData.posDeltaY = centerPosY - origCenterPosY;
            _simController->shallowUpdateSelectedObjects(updateData);
            _editorModel->update();
        }

        if (centerVelX != origCenterVelX || centerVelY != origCenterVelY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.velDeltaX = centerVelX - origCenterVelX;
            updateData.velDeltaY = centerVelY - origCenterVelY;
            _simController->shallowUpdateSelectedObjects(updateData);
            _editorModel->update();
        }

        if (_angle != origAngle) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.angleDelta = _angle - origAngle;
            _simController->shallowUpdateSelectedObjects(updateData);
            _editorModel->update();
        }

        if (_angularVel != origAngularVel) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _editorModel->isRolloutToClusters();
            updateData.angularVelDelta = _angularVel - origAngularVel;
            _simController->shallowUpdateSelectedObjects(updateData);
            _editorModel->update();
        }
        ImGui::EndDisabled();


        AlienImGui::Group("Color");
        if (colorButton("    ##color1", Const::IndividualCellColor1)) {
            _simController->colorSelectedObjects(0, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(0);
        }
        ImGui::SameLine();
        if (colorButton("    ##color2", Const::IndividualCellColor2)) {
            _simController->colorSelectedObjects(1, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(1);
        }
        ImGui::SameLine();
        if (colorButton("    ##color3", Const::IndividualCellColor3)) {
            _simController->colorSelectedObjects(2, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(2);
        }
        ImGui::SameLine();
        if (colorButton("    ##color4", Const::IndividualCellColor4)) {
            _simController->colorSelectedObjects(3, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(3);
        }
        ImGui::SameLine();
        if (colorButton("    ##color5", Const::IndividualCellColor5)) {
            _simController->colorSelectedObjects(4, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(4);
        }
        ImGui::SameLine();
        if (colorButton("    ##color6", Const::IndividualCellColor6)) {
            _simController->colorSelectedObjects(5, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(5);
        }
        ImGui::SameLine();
        if (colorButton("    ##color7", Const::IndividualCellColor7)) {
            _simController->colorSelectedObjects(6, _editorModel->isRolloutToClusters());
            _editorModel->setDefaultColorCode(6);
        }
        AlienImGui::Group("Tools");
        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
        if (ImGui::Button(ICON_FA_WIND)) {
            _simController->uniformVelocitiesForSelectedObjects(_editorModel->isRolloutToClusters());
        }
        ImGui::EndDisabled();
        AlienImGui::Tooltip("Make uniform velocities");

        ImGui::SameLine();
        ImGui::BeginDisabled(_editorModel->isCellSelectionEmpty());
        if (ImGui::Button(ICON_FA_BALANCE_SCALE)) {
            _simController->relaxSelectedObjects(_editorModel->isRolloutToClusters());
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
        AlienImGui::Tooltip("Make indestructible");

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_UNLINK)) {
            onSetBarrier(false);
        }
        AlienImGui::Tooltip("Make destructible");
        ImGui::EndDisabled();

        _lastSelection = selection;
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    auto rolloutToClusters = _editorModel->isRolloutToClusters();
    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Roll out changes to cell networks"), rolloutToClusters)) {
        _editorModel->setRolloutToClusters(rolloutToClusters);
        _angle = 0;
        _angularVel = 0;
    }
    ImGui::SameLine();
    AlienImGui::HelpMarker("If turned on, all changes made in this window or with the mouse cursor are applied to the cell networks of the selected cell.\n"
                           "If this option is disabled, the changes will be applied only to the selected cells. In this case, the connections between the cells and the neighboring cells are recalculated when the positions are changed.\n"
                           "If you hold down the SHIFT key, this toggle button is temporarily turned off.");
}

void _PatternEditorWindow::onOpenPattern()
{
    GenericFileDialogs::getInstance().showOpenFileDialog(
        "Open pattern", "Pattern file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();
            ClusteredDataDescription content;
            if (Serializer::deserializeContentFromFile(content, firstFilename.string())) {
                auto center = _viewport->getCenterInWorldPos();
                content.setCenter(center);
                _simController->addAndSelectSimulationData(DataDescription(content));
                _editorModel->update();
            } else {
                MessageDialog::getInstance().information("Open pattern", "The selected file could not be opened.");
            }
        });
}

void _PatternEditorWindow::onSavePattern()
{
    GenericFileDialogs::getInstance().showSaveFileDialog(
        "Save pattern", "Pattern file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            auto content = _simController->getSelectedClusteredSimulationData(_editorModel->isRolloutToClusters());
            if (!Serializer::serializeContentToFile(firstFilename.string(), content)) {
                MessageDialog::getInstance().information("Save pattern", "The selected pattern could not be saved to the specified file.");
            }
        });
}

bool _PatternEditorWindow::isObjectInspectionPossible() const
{
    return !_editorModel->isSelectionEmpty();
}

bool _PatternEditorWindow::isGenomeInspectionPossible() const
{
    return !_editorModel->isSelectionEmpty();
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
    DescriptionEditService::generateNewCreatureIds(data);
    _simController->addAndSelectSimulationData(data);
    _editorModel->update();
}

bool _PatternEditorWindow::isDeletingPossible() const
{
    return !_editorModel->isSelectionEmpty() && !_editorModel->areEntitiesInspected();
}

void _PatternEditorWindow::onDelete()
{
    _simController->removeSelectedObjects(_editorModel->isRolloutToClusters());
    _editorModel->update();
}

void _PatternEditorWindow::onGenerateExecutionOrderNumbers()
{
    auto dataWithClusters = _simController->getSelectedSimulationData(true);
    auto dataWithoutClusters = _simController->getSelectedSimulationData(false);
    std::unordered_set<uint64_t> cellIds = dataWithoutClusters.getCellIds();

    auto parameters = _simController->getSimulationParameters();
    DescriptionEditService::generateExecutionOrderNumbers(dataWithClusters, cellIds, parameters.cellNumExecutionOrderNumbers);

    _simController->removeSelectedObjects(true);
    _simController->addAndSelectSimulationData(dataWithClusters);
}

void _PatternEditorWindow::onMakeSticky()
{
    _simController->makeSticky(_editorModel->isRolloutToClusters());
}

void _PatternEditorWindow::onRemoveStickiness()
{
    _simController->removeStickiness(_editorModel->isRolloutToClusters());
}

void _PatternEditorWindow::onSetBarrier(bool value)
{
    _simController->setBarrier(value, _editorModel->isRolloutToClusters());
}

bool _PatternEditorWindow::colorButton(std::string id, uint32_t cellColor)
{
    ImGui::PushID(id.c_str());
    auto result = AlienImGui::ColorField(cellColor);
    ImGui::PopID();
    return result;
}

bool _PatternEditorWindow::hasSelectionChanged(SelectionShallowData const& selection) const
{
    if(!_lastSelection) {
        return false;
    }
    return _lastSelection->numCells != selection.numCells || _lastSelection->numParticles != selection.numParticles;
}
