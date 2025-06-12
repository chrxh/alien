#include "NodeEditorWidget.h"

#include "AlienGui.h"
#include "CreatureTabEditData.h"
#include "CreatureTabLayoutData.h"
#include "LoginDialog.h"
#include "NeuralNetWidget.h"

namespace
{
    auto constexpr HeaderLeftColumnWidth = 140.0f;
}

NodeEditorWidget _NodeEditorWidget::create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(editData, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        auto nodeIndex = _editData->getSelectedNodeIndex();
        if (nodeIndex.has_value()) {
            ImGui::PushID(_editData->selectedGeneIndex.value());
            ImGui::PushID(nodeIndex.value());
            processNodeAttributes();

            AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->neuralNetEditorHeight);

            processNeuralNetEditor();
            ImGui::PopID();
            ImGui::PopID();
        } else {
            processNoSelection();
        }
    }
    ImGui::EndChild();
}

_NodeEditorWidget::_NodeEditorWidget(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
    : _editData(editData)
    , _layoutData(layoutData)
{
    _neuralNetWidget = _NeuralNetWidget::create();
}

namespace
{
    CellTypeGenomeDescription_New createEmptyCellTypeGenomeDescription(CellTypeGenome cellType)
    {
        switch (cellType) {
        case CellTypeGenome_Base:
            return BaseGenomeDescription();
        case CellTypeGenome_Depot:
            return DepotGenomeDescription();
        case CellTypeGenome_Constructor:
            return ConstructorGenomeDescription_New();
        case CellTypeGenome_Sensor:
            return SensorGenomeDescription();
        case CellTypeGenome_Oscillator:
            return OscillatorGenomeDescription();
        case CellTypeGenome_Attacker:
            return AttackerGenomeDescription();
        case CellTypeGenome_Injector:
            return InjectorGenomeDescription_New();
        case CellTypeGenome_Muscle:
            return MuscleGenomeDescription();
        case CellTypeGenome_Defender:
            return DefenderGenomeDescription();
        case CellTypeGenome_Reconnector:
            return ReconnectorGenomeDescription();
        case CellTypeGenome_Detonator:
            return DetonatorGenomeDescription();
        default:
            CHECK(false);
        }
    }
}

void _NodeEditorWidget::processNodeAttributes()
{
    AlienGui::Group("Selected node");

    auto rightColumnWidth = scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderLeftColumnWidth));
    if (ImGui::BeginChild("NodeData", ImVec2(0, -_layoutData->neuralNetEditorHeight), 0)) {
        auto& gene = _editData->getSelectedGeneRef();
        auto& node = _editData->getSelectedNodeRef();
        auto nodeType = node.getCellType();

        if (AlienGui::Combo(AlienGui::ComboParameters().name("Type").values(Const::CellTypeGenomeStrings).textWidth(rightColumnWidth), nodeType)) {
            node._cellTypeData = createEmptyCellTypeGenomeDescription(nodeType);
        }

        auto nodeIndex = _editData->getSelectedNodeIndex();
        if (nodeIndex != 0 && nodeIndex != gene._nodes.size() - 1) {
            if (AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Angle").textWidth(rightColumnWidth).format("%.1f"), node._referenceAngle)) {
                gene._shape = ConstructionShape_Custom;
            }
        } else {
            std::string text = "-";
            AlienGui::InputText(AlienGui::InputTextParameters().name("Angle").textWidth(rightColumnWidth).readOnly(true), text);
        }

        if (nodeIndex != 0) {
            auto numRequiredAdditionalConnections = node._numRequiredAdditionalConnections + 1;
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Prev nodes connections").textWidth(rightColumnWidth), node._numRequiredAdditionalConnections)) {
                gene._shape = ConstructionShape_Custom;
            }
            node._numRequiredAdditionalConnections = numRequiredAdditionalConnections - 1;
        } else {
            std::string text = "-";
            AlienGui::InputText(AlienGui::InputTextParameters().name("Prev nodes connections").textWidth(rightColumnWidth).readOnly(true), text);
        }

        AlienGui::Checkbox(
            AlienGui::CheckboxParameters().name("Signal routing restriction").textWidth(rightColumnWidth), node._signalRoutingRestriction._active);

        AlienGui::BeginIndent();

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Signal base angle")
                .format("%.1f")
                .step(0.5f)
                .readOnly(!node._signalRoutingRestriction._active)
                .textWidth(rightColumnWidth),
            node._signalRoutingRestriction._baseAngle);

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Signal opening angle")
                .format("%.1f")
                .step(0.5f)
                .readOnly(!node._signalRoutingRestriction._active)
                .textWidth(rightColumnWidth),
            node._signalRoutingRestriction._openingAngle);

        AlienGui::EndIndent();

        AlienGui::ComboColor(AlienGui::ComboColorParameters().name("Color").textWidth(rightColumnWidth), node._color);
    }
    ImGui::EndChild();
}

void _NodeEditorWidget::processNoSelection()
{
    AlienGui::Group("Selected node");
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        AlienGui::DisabledField();
        auto text = "No node is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}

void _NodeEditorWidget::processNeuralNetEditor()
{
    AlienGui::MoveTickUp();
    AlienGui::MoveTickUp();
    AlienGui::Group("Neural net");

    auto& node = _editData->getSelectedNodeRef();
    _neuralNetWidget->process(node._neuralNetwork._weights, node._neuralNetwork._biases, node._neuralNetwork._activationFunctions);
}
