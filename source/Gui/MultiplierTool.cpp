#include "MultiplierTool.h"

#include <imgui.h>

#include <Fonts/AlienIconFont.h>
#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto const ModeText = std::unordered_map<MultiplierMode, std::string>{
        {MultiplierMode_Grid, "Grid multiplier"},
        {MultiplierMode_Random, "Random multiplier"},
    };
    auto constexpr RightColumnWidth = 170.0f;
    auto constexpr GridLabelColumnWidth = 130.0f;
    auto constexpr PreviewFrameThickness = 1.0f;
    auto constexpr PreviewFramePadding = 3.0f;
    auto constexpr MaxNumPreviewFrames = 2500;
}

void MultiplierTool::init()
{
    auto& settings = GlobalSettings::get();
    _mode = settings.getValue("editors.multiplier.mode", _mode);
    _gridParameters._horizontalNumber = settings.getValue("editors.multiplier.grid.horizontal number", _gridParameters._horizontalNumber);
    _gridParameters._horizontalDistance = settings.getValue("editors.multiplier.grid.horizontal distance", _gridParameters._horizontalDistance);
    _gridParameters._horizontalAngleInc = settings.getValue("editors.multiplier.grid.horizontal angle inc", _gridParameters._horizontalAngleInc);
    _gridParameters._horizontalVelXinc = settings.getValue("editors.multiplier.grid.horizontal vel x inc", _gridParameters._horizontalVelXinc);
    _gridParameters._horizontalVelYinc = settings.getValue("editors.multiplier.grid.horizontal vel y inc", _gridParameters._horizontalVelYinc);
    _gridParameters._horizontalAngularVelInc =
        settings.getValue("editors.multiplier.grid.horizontal angular vel inc", _gridParameters._horizontalAngularVelInc);
    _gridParameters._verticalNumber = settings.getValue("editors.multiplier.grid.vertical number", _gridParameters._verticalNumber);
    _gridParameters._verticalDistance = settings.getValue("editors.multiplier.grid.vertical distance", _gridParameters._verticalDistance);
    _gridParameters._verticalAngleInc = settings.getValue("editors.multiplier.grid.vertical angle inc", _gridParameters._verticalAngleInc);
    _gridParameters._verticalVelXinc = settings.getValue("editors.multiplier.grid.vertical vel x inc", _gridParameters._verticalVelXinc);
    _gridParameters._verticalVelYinc = settings.getValue("editors.multiplier.grid.vertical vel y inc", _gridParameters._verticalVelYinc);
    _gridParameters._verticalAngularVelInc = settings.getValue("editors.multiplier.grid.vertical angular vel inc", _gridParameters._verticalAngularVelInc);
    _randomParameters._number = settings.getValue("editors.multiplier.random.number", _randomParameters._number);
    _randomParameters._minAngle = settings.getValue("editors.multiplier.random.min angle", _randomParameters._minAngle);
    _randomParameters._maxAngle = settings.getValue("editors.multiplier.random.max angle", _randomParameters._maxAngle);
    _randomParameters._minVelX = settings.getValue("editors.multiplier.random.min vel x", _randomParameters._minVelX);
    _randomParameters._maxVelX = settings.getValue("editors.multiplier.random.max vel x", _randomParameters._maxVelX);
    _randomParameters._minVelY = settings.getValue("editors.multiplier.random.min vel y", _randomParameters._minVelY);
    _randomParameters._maxVelY = settings.getValue("editors.multiplier.random.max vel y", _randomParameters._maxVelY);
    _randomParameters._minAngularVel = settings.getValue("editors.multiplier.random.min angular vel", _randomParameters._minAngularVel);
    _randomParameters._maxAngularVel = settings.getValue("editors.multiplier.random.max angular vel", _randomParameters._maxAngularVel);
    _randomParameters._overlappingCheck = settings.getValue("editors.multiplier.random.overlapping check", _randomParameters._overlappingCheck);
}

void MultiplierTool::shutdown()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("editors.multiplier.mode", _mode);
    settings.setValue("editors.multiplier.grid.horizontal number", _gridParameters._horizontalNumber);
    settings.setValue("editors.multiplier.grid.horizontal distance", _gridParameters._horizontalDistance);
    settings.setValue("editors.multiplier.grid.horizontal angle inc", _gridParameters._horizontalAngleInc);
    settings.setValue("editors.multiplier.grid.horizontal vel x inc", _gridParameters._horizontalVelXinc);
    settings.setValue("editors.multiplier.grid.horizontal vel y inc", _gridParameters._horizontalVelYinc);
    settings.setValue("editors.multiplier.grid.horizontal angular vel inc", _gridParameters._horizontalAngularVelInc);
    settings.setValue("editors.multiplier.grid.vertical number", _gridParameters._verticalNumber);
    settings.setValue("editors.multiplier.grid.vertical distance", _gridParameters._verticalDistance);
    settings.setValue("editors.multiplier.grid.vertical angle inc", _gridParameters._verticalAngleInc);
    settings.setValue("editors.multiplier.grid.vertical vel x inc", _gridParameters._verticalVelXinc);
    settings.setValue("editors.multiplier.grid.vertical vel y inc", _gridParameters._verticalVelYinc);
    settings.setValue("editors.multiplier.grid.vertical angular vel inc", _gridParameters._verticalAngularVelInc);
    settings.setValue("editors.multiplier.random.number", _randomParameters._number);
    settings.setValue("editors.multiplier.random.min angle", _randomParameters._minAngle);
    settings.setValue("editors.multiplier.random.max angle", _randomParameters._maxAngle);
    settings.setValue("editors.multiplier.random.min vel x", _randomParameters._minVelX);
    settings.setValue("editors.multiplier.random.max vel x", _randomParameters._maxVelX);
    settings.setValue("editors.multiplier.random.min vel y", _randomParameters._minVelY);
    settings.setValue("editors.multiplier.random.max vel y", _randomParameters._maxVelY);
    settings.setValue("editors.multiplier.random.min angular vel", _randomParameters._minAngularVel);
    settings.setValue("editors.multiplier.random.max angular vel", _randomParameters._maxAngularVel);
    settings.setValue("editors.multiplier.random.overlapping check", _randomParameters._overlappingCheck);
}

void MultiplierTool::process() {}

void MultiplierTool::processContent()
{
    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("Multiplier").bottomSeparator(false),
        {AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_GRID).name(ModeText.at(MultiplierMode_Grid)).selected(_mode == MultiplierMode_Grid).action([&] {
                 _mode = MultiplierMode_Grid;
             })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_RANDOM).name(ModeText.at(MultiplierMode_Random)).selected(_mode == MultiplierMode_Random).action([&] {
                 _mode = MultiplierMode_Random;
             }))});

    AlienGui::Group(AlienGui::GroupParameters().text(ModeText.at(_mode)));
    if (_mode == MultiplierMode_Grid) {
        processGridPanel();
        processGridPreview();
    } else {
        processRandomPanel();
    }

    AlienGui::Separator();
    auto const& selection = EditorModel::get().getSelectionShallowData();
    auto multiplied = _selectionDataAfterMultiplication.has_value() && _selectionDataAfterMultiplication->compareSizes(selection);
    ImGui::BeginDisabled(multiplied);
    if (AlienGui::Button("Build")) {
        onBuild();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!multiplied);
    if (AlienGui::Button("Undo")) {
        onUndo();
    }
    ImGui::EndDisabled();

    validateAndCorrect();
}

void MultiplierTool::processGridPanel()
{
    if (!ImGui::BeginTable("##grid", 3, ImGuiTableFlags_SizingStretchSame)) {
        return;
    }
    ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, scale(GridLabelColumnWidth));
    ImGui::TableSetupColumn(ICON_FA_ARROW_RIGHT "  Horizontal");
    ImGui::TableSetupColumn(ICON_FA_ARROW_DOWN "  Vertical");
    ImGui::TableHeadersRow();

    auto intRow = [](char const* label, int& horizontal, int& vertical) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::PushID(label);
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputInt("##horizontal", &horizontal, 0);
        ImGui::TableSetColumnIndex(2);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputInt("##vertical", &vertical, 0);
        ImGui::PopID();
    };
    auto floatRow = [](char const* label, float& horizontal, float& vertical, char const* format) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::PushID(label);
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##horizontal", &horizontal, 0, 0, format);
        ImGui::TableSetColumnIndex(2);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##vertical", &vertical, 0, 0, format);
        ImGui::PopID();
    };

    intRow("Number of copies", _gridParameters._horizontalNumber, _gridParameters._verticalNumber);
    floatRow("Distance", _gridParameters._horizontalDistance, _gridParameters._verticalDistance, "%.1f");
    floatRow("Angle increment", _gridParameters._horizontalAngleInc, _gridParameters._verticalAngleInc, "%.1f");
    floatRow("Velocity X increment", _gridParameters._horizontalVelXinc, _gridParameters._verticalVelXinc, "%.2f");
    floatRow("Velocity Y increment", _gridParameters._horizontalVelYinc, _gridParameters._verticalVelYinc, "%.2f");
    floatRow("Angular vel. increment", _gridParameters._horizontalAngularVelInc, _gridParameters._verticalAngularVelInc, "%.1f");

    ImGui::EndTable();
}

void MultiplierTool::processRandomPanel()
{
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Number of copies").textWidth(RightColumnWidth), _randomParameters._number);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Min angle").textWidth(RightColumnWidth).format("%.1f"), _randomParameters._minAngle);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Max angle").textWidth(RightColumnWidth).format("%.1f"), _randomParameters._maxAngle);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Min velocity X").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._minVelX);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max velocity X").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._maxVelX);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Min velocity Y").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._minVelY);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max velocity Y").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._maxVelY);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Min angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f), _randomParameters._minAngularVel);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f), _randomParameters._maxAngularVel);
    AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Overlapping check").textWidth(RightColumnWidth), &_randomParameters._overlappingCheck);
}

void MultiplierTool::processGridPreview() const
{
    if (_gridParameters._horizontalNumber * _gridParameters._verticalNumber > MaxNumPreviewFrames) {
        return;
    }
    auto bounds = EditorModel::get().getSelectionBounds(true);
    auto zoom = Viewport::get().getZoomFactor();
    auto borderlessRendering = _SimulationFacade::get()->getSimulationParameters().borderlessRendering.value;
    auto viewCenter = Viewport::get().mapWorldToViewPosition(bounds.center, borderlessRendering);
    auto padding = scale(PreviewFramePadding);
    auto topLeft = viewCenter + (bounds.topLeft - bounds.center) * zoom - RealVector2D{padding, padding};
    auto bottomRight = viewCenter + (bounds.bottomRight - bounds.center) * zoom + RealVector2D{padding, padding};

    auto drawList = ImGui::GetBackgroundDrawList();
    for (int i = 0; i < _gridParameters._horizontalNumber; ++i) {
        for (int j = 0; j < _gridParameters._verticalNumber; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }
            auto offset = RealVector2D{toFloat(i) * _gridParameters._horizontalDistance, toFloat(j) * _gridParameters._verticalDistance} * zoom;
            drawList->AddRect(
                {topLeft.x + offset.x, topLeft.y + offset.y},
                {bottomRight.x + offset.x, bottomRight.y + offset.y},
                Const::MultiplierPreviewColor,
                scale(2.0f),
                0,
                scale(PreviewFrameThickness));
        }
    }
}

void MultiplierTool::validateAndCorrect()
{
    _gridParameters._horizontalNumber = std::max(1, _gridParameters._horizontalNumber);
    _gridParameters._horizontalDistance = std::max(0.0f, _gridParameters._horizontalDistance);
    _gridParameters._verticalNumber = std::max(1, _gridParameters._verticalNumber);
    _gridParameters._verticalDistance = std::max(0.0f, _gridParameters._verticalDistance);
    _randomParameters._number = std::max(1, _randomParameters._number);
    _randomParameters._maxAngle = std::max(_randomParameters._minAngle, _randomParameters._maxAngle);
    _randomParameters._maxVelX = std::max(_randomParameters._minVelX, _randomParameters._maxVelX);
    _randomParameters._maxVelY = std::max(_randomParameters._minVelY, _randomParameters._maxVelY);
    _randomParameters._maxAngularVel = std::max(_randomParameters._minAngularVel, _randomParameters._maxAngularVel);
}

namespace
{
    void replaceSelection(ContentDesc&& content)
    {
        _SimulationFacade::get()->removeSelectedObjects(true);
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    }
}

void MultiplierTool::onBuild()
{
    _origSelection = _SimulationFacade::get()->getSelectedSimulationData(true);
    if (_mode == MultiplierMode_Grid) {
        replaceSelection(MultiplierService::get().multiplyInGrid(_origSelection, _gridParameters));
    } else {
        auto parameters = MultiplierService::RandomParameters(_randomParameters).maxDelta(_SimulationFacade::get()->getWorldSize());
        auto multiplication = MultiplierService::get().multiplyRandomly(_origSelection, parameters);
        replaceSelection(std::move(multiplication.content));
        if (!multiplication.overlappingCheckSuccessful) {
            GenericMessageDialog::get().information("Random multiplication", "Non-overlapping copies could not be created.");
        }
    }
    EditorModel::get().update();
    _selectionDataAfterMultiplication = EditorModel::get().getSelectionShallowData();
}

void MultiplierTool::onUndo()
{
    replaceSelection(ContentDesc(_origSelection));
    EditorModel::get().update();
    _selectionDataAfterMultiplication = std::nullopt;
}
