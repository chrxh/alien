#include "NodeEditorWidget.h"

#include "AlienGui.h"
#include "CreatureTabEditData.h"
#include "LoginDialog.h"

namespace
{
    auto constexpr TotalWidth = 400.0f;
    auto constexpr RightColumnWidth = 120.0f;
}

NodeEditorWidget _NodeEditorWidget::create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(editData, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        if (_editData->getSelectedNodeIndex()) {
            processNodeAttributes();
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

    auto availableWidth = scaleInverse(ImGui::GetContentRegionAvail().x);
    auto width = scale(std::min(availableWidth, TotalWidth));

    if (ImGui::BeginChild("NodeData", ImVec2(width, 0), 0)) {
        auto& gene = _editData->getSelectedGeneRef();
        auto& node = _editData->getSelectedNodeRef();
        auto nodeType = node.getCellType();

        if (AlienGui::Combo(
            AlienGui::ComboParameters().name("Type").values(Const::CellTypeGenomeStrings).textWidth(RightColumnWidth), nodeType)) {
            node._cellTypeData = createEmptyCellTypeGenomeDescription(nodeType);
        }

        auto nodeIndex = _editData->getSelectedNodeIndex();
        if (nodeIndex != 0 && nodeIndex != gene._nodes.size() - 1) {
            if (AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Angle").textWidth(RightColumnWidth).format("%.1f"),
                    node._referenceAngle)) {
                gene._shape = ConstructionShape_Custom;
            }
        }

        AlienGui::ComboColor(AlienGui::ComboColorParameters().name("Color").textWidth(RightColumnWidth), node._color);
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
