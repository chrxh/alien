#include "NodeEditorWidget.h"

#include <boost/range/adaptors.hpp>

#include "AlienGui.h"
#include "CreatureTabEditData.h"
#include "CreatureTabLayoutData.h"
#include "LoginDialog.h"
#include "NeuralNetEditorWidget.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 170.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
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
    _neuralNetWidget = _NeuralNetEditorWidget::create();
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
            return SensorGenomeDescription_New();
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

    auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));
    if (ImGui::BeginChild("NodeData", ImVec2(0, -_layoutData->neuralNetEditorHeight), 0, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto& gene = _editData->getSelectedGeneRef();
        auto& node = _editData->getSelectedNodeRef();
        auto nodeType = node.getCellType();

        // Type
        if (AlienGui::Combo(AlienGui::ComboParameters().name("Type").values(Const::CellTypeGenomeStrings).textWidth(rightColumnWidth), nodeType)) {
            node._cellTypeData = createEmptyCellTypeGenomeDescription(nodeType);
        }

        if (nodeType == CellTypeGenome_Base) {

        } else if (nodeType == CellTypeGenome_Depot) {

        } else if (nodeType == CellTypeGenome_Constructor) {

            AlienGui::BeginIndent();

            // Activation interval
            auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Auto activation interval").textWidth(rightColumnWidth), constructor._autoTriggerInterval);

            // Gene index
            std::vector<std::string> genes;
            for (auto const& [index, gene] : _editData->genome._genes | boost::adaptors::indexed(0)) {
                auto text = "No. " + std::to_string(index + 1);
                if (index == 0) {
                    text += " (reproduction)";
                }
                genes.emplace_back(text);
            }
            AlienGui::Combo(AlienGui::ComboParameters().name("Gene").values(genes).textWidth(rightColumnWidth), constructor._constructGeneIndex);

            // Construction activation time
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Offspring activation time").textWidth(rightColumnWidth), constructor._constructionActivationTime);

            // Construction angle
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Construction angle").format("%.1f").textWidth(rightColumnWidth), constructor._constructionAngle);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Sensor) {

            AlienGui::BeginIndent();

            // Activation mode
            auto& sensor = std::get<SensorGenomeDescription_New>(node._cellTypeData);
            AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Auto activation interval").textWidth(rightColumnWidth), sensor._autoTriggerInterval);

            // Minimum density
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Min density").format("%.2f").textWidth(rightColumnWidth), sensor._minDensity);

            // Minimum range
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Min range").textWidth(rightColumnWidth), sensor._minRange);

            // Maximum range
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Max range").textWidth(rightColumnWidth), sensor._maxRange);

            // Scan color
            AlienGui::ComboOptionalColor(AlienGui::ComboColorParameters().name("Scan color").textWidth(rightColumnWidth), sensor._restrictToColor);

            // Scan mutants
            AlienGui::Combo(
                AlienGui::ComboParameters().name("Scan mutants").values(Const::SensorRestrictToMutantStrings).textWidth(rightColumnWidth),
                sensor._restrictToMutants);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Oscillator) {
        }

        // Angle
        auto nodeIndex = _editData->getSelectedNodeIndex();
        if (AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Angle").textWidth(rightColumnWidth).format("%.1f"), node._referenceAngle)) {
            if (nodeIndex.value() != 0 && nodeIndex != gene._nodes.size() - 1) {
                gene._shape = ConstructionShape_Custom;
            }
        }

        // Previous nodes connections
        if (nodeIndex != 0) {
            auto numRequiredAdditionalConnections = node._numRequiredAdditionalConnections + 1;
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Prev nodes connections").textWidth(rightColumnWidth), numRequiredAdditionalConnections)) {
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
    AlienGui::Group("Neural network");

    auto& node = _editData->getSelectedNodeRef();
    _neuralNetWidget->process(node._neuralNetwork._weights, node._neuralNetwork._biases, node._neuralNetwork._activationFunctions);
}
