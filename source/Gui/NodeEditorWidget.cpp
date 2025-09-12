#include "NodeEditorWidget.h"

#include <boost/range/adaptors.hpp>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "LoginDialog.h"
#include "NeuralNetEditorWidget.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 170.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
}


NodeEditorWidget _NodeEditorWidget::create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
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

_NodeEditorWidget::_NodeEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
    : _editData(editData)
    , _layoutData(layoutData)
{
    _neuralNetWidget = _NeuralNetEditorWidget::create();
}

namespace
{
    CellTypeGenomeDescription createCellTypeGenomeDescription(CellTypeGenome cellType)
    {
        switch (cellType) {
        case CellTypeGenome_Base:
            return BaseGenomeDescription();
        case CellTypeGenome_Depot:
            return DepotGenomeDescription();
        case CellTypeGenome_Constructor:
            return ConstructorGenomeDescription();
        case CellTypeGenome_Sensor:
            return SensorGenomeDescription();
        case CellTypeGenome_Generator:
            return GeneratorGenomeDescription();
        case CellTypeGenome_Attacker:
            return AttackerGenomeDescription();
        case CellTypeGenome_Injector:
            return InjectorGenomeDescription();
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

    MuscleModeGenomeDescription createMuscleModeGenomeDescription(MuscleMode mode)
    {
        switch (mode) {
        case MuscleMode_AutoBending:
            return AutoBendingGenomeDescription();
        case MuscleMode_ManualBending:
            return ManualBendingGenomeDescription();
        case MuscleMode_AngleBending:
            return AngleBendingGenomeDescription();
        case MuscleMode_AutoCrawling:
            return AutoCrawlingGenomeDescription();
        case MuscleMode_ManualCrawling:
            return ManualCrawlingGenomeDescription();
        case MuscleMode_DirectMovement:
            return DirectMovementGenomeDescription();
        default:
            CHECK(false);
        }
    }
}

void _NodeEditorWidget::processNodeAttributes()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Selected node").highlighted(true));

    auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));
    if (ImGui::BeginChild("NodeData", ImVec2(0, -_layoutData->neuralNetEditorHeight), 0, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto& gene = _editData->getSelectedGeneRef();
        auto& node = _editData->getSelectedNodeRef();

        // Angle
        auto nodeIndex = _editData->getSelectedNodeIndex();
        if (AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Angle").textWidth(rightColumnWidth).format("%.1f"), node._referenceAngle)) {
            if (nodeIndex.value() != 0 && nodeIndex != gene._nodes.size() - 1) {
                gene._shape = ConstructorShape_Custom;
            }
        }

        // Previous nodes connections
        if (nodeIndex != 0) {
            auto numAdditionalConnections = node._numAdditionalConnections + 1;
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Prev nodes connections").textWidth(rightColumnWidth), numAdditionalConnections)) {
                gene._shape = ConstructorShape_Custom;
            }
            node._numAdditionalConnections = std::max(numAdditionalConnections - 1, 0);
        }

        AlienGui::Checkbox(
            AlienGui::CheckboxParameters().name("Signal restriction").textWidth(rightColumnWidth), node._signalRestriction._active);

        AlienGui::BeginIndent();

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Signal base angle")
                .format("%.1f")
                .step(0.5f)
                .readOnly(!node._signalRestriction._active)
                .textWidth(rightColumnWidth),
            node._signalRestriction._baseAngle);

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Signal opening angle")
                .format("%.1f")
                .step(0.5f)
                .readOnly(!node._signalRestriction._active)
                .textWidth(rightColumnWidth),
            node._signalRestriction._openingAngle);

        AlienGui::EndIndent();

        AlienGui::ComboColor(AlienGui::ComboColorParameters().name("Color").textWidth(rightColumnWidth), node._color);

        // Type
        auto nodeType = node.getCellType();
        if (AlienGui::Combo(AlienGui::ComboParameters().name("Type").values(Const::CellTypeGenomeStrings).textWidth(rightColumnWidth), nodeType)) {
            node._cellTypeData = createCellTypeGenomeDescription(nodeType);
        }
        if (nodeType == CellTypeGenome_Base) {
        } else if (nodeType == CellTypeGenome_Depot) {
        } else if (nodeType == CellTypeGenome_Constructor) {

            AlienGui::BeginIndent();

            // Gene index
            auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
            std::vector<std::string> genes;
            for (auto const& [index, gene] : _editData->genome._genes | boost::adaptors::indexed(0)) {
                auto text = "No. " + std::to_string(index + 1);
                if (index == 0) {
                    text += " (root)";
                }
                genes.emplace_back(text);
            }
            AlienGui::Combo(AlienGui::ComboParameters().name("Gene").values(genes).textWidth(rightColumnWidth), constructor._geneIndex);

            // Auto activation interval
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Auto activation interval").textWidth(rightColumnWidth), constructor._autoTriggerInterval);

            // Construction activation time
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Offspring activation time").textWidth(rightColumnWidth), constructor._constructionActivationTime);

            // Construction angle
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Construction angle").textWidth(rightColumnWidth).format("%.1f"), constructor._constructionAngle);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Sensor) {

            AlienGui::BeginIndent();

            // Auto activation interval
            auto& sensor = std::get<SensorGenomeDescription>(node._cellTypeData);
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Auto activation interval").textWidth(rightColumnWidth), sensor._autoTriggerInterval);

            // Minimum density
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Min density").format("%.2f").textWidth(rightColumnWidth), sensor._minDensity);

            // Minimum range
            AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Min range").textWidth(rightColumnWidth), sensor._minRange);

            // Maximum range
            AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Max range").textWidth(rightColumnWidth), sensor._maxRange);

            // Scan color
            AlienGui::ComboOptionalColor(AlienGui::ComboColorParameters().name("Scan color").textWidth(rightColumnWidth), sensor._restrictToColor);

            // Scan mutants
            AlienGui::Combo(
                AlienGui::ComboParameters().name("Scan mutants").values(Const::SensorRestrictToMutantStrings).textWidth(rightColumnWidth),
                sensor._restrictToCreatures);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Generator) {

            AlienGui::BeginIndent();

            // Activation interval
            auto& generator = std::get<GeneratorGenomeDescription>(node._cellTypeData);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Activation interval").textWidth(rightColumnWidth), generator._autoTriggerInterval);

            // Pulse type
            AlienGui::Combo(
                AlienGui::ComboParameters().name("Pulse type").values({"Positive", "Alternation"}).textWidth(rightColumnWidth), generator._pulseType);

            if (generator._pulseType == GeneratorPulseType_Alternation) {

                AlienGui::BeginIndent();

                // Pulses per phase
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Pulses per phase").textWidth(rightColumnWidth), generator._alternationInterval);

                AlienGui::EndIndent();
            }

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Attacker) {
        } else if (nodeType == CellTypeGenome_Injector) {

            AlienGui::BeginIndent();

            // Mode
            auto& injector = std::get<InjectorGenomeDescription>(node._cellTypeData);
            AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::InjectorModeStrings).textWidth(rightColumnWidth), injector._mode);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Muscle) {

            AlienGui::BeginIndent();

            // Mode
            auto& muscle = std::get<MuscleGenomeDescription>(node._cellTypeData);
            auto mode = muscle.getMode();
            if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::MuscleModeStrings).textWidth(rightColumnWidth), mode)) {
                muscle._mode = createMuscleModeGenomeDescription(mode);
            }

            if (mode == MuscleMode_AutoBending) {
                AlienGui::BeginIndent();

                // Max angle deviation
                auto& autoBending = std::get<AutoBendingGenomeDescription>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    autoBending._maxAngleDeviation);

                // Front back ratio
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    autoBending._frontBackVelRatio);

                AlienGui::EndIndent();

            } else if (mode == MuscleMode_ManualBending) {
                AlienGui::BeginIndent();

                // Max angle deviation
                auto& manualBending = std::get<ManualBendingGenomeDescription>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    manualBending._maxAngleDeviation);

                // Front back ratio
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    manualBending._frontBackVelRatio);

                AlienGui::EndIndent();

            } else if (mode == MuscleMode_AngleBending) {
                AlienGui::BeginIndent();

                // Max angle deviation
                auto& angleBending = std::get<AngleBendingGenomeDescription>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    angleBending._maxAngleDeviation);

                // Front back ratio
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    angleBending._frontBackVelRatio);

                AlienGui::EndIndent();

            } else if (mode == MuscleMode_AutoCrawling) {
                AlienGui::BeginIndent();

                // Max angle deviation
                auto& autoCrawling = std::get<AutoCrawlingGenomeDescription>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max distance deviation").format("%.2f").step(0.01f).textWidth(rightColumnWidth),
                    autoCrawling._maxDistanceDeviation);

                // Front back ratio
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    autoCrawling._frontBackVelRatio);

                AlienGui::EndIndent();

            } else if (mode == MuscleMode_ManualCrawling) {
                AlienGui::BeginIndent();

                // Max angle deviation
                auto& manualCrawling = std::get<ManualCrawlingGenomeDescription>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max distance deviation").format("%.2f").step(0.01f).textWidth(rightColumnWidth),
                    manualCrawling._maxDistanceDeviation);

                // Front back ratio
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                    manualCrawling._frontBackVelRatio);

                AlienGui::EndIndent();

            } else if (mode == MuscleMode_DirectMovement) {
            }

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Defender) {

            AlienGui::BeginIndent();

            // Defender mode
            auto& defender = std::get<DefenderGenomeDescription>(node._cellTypeData);
            AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::DefenderModeStrings).textWidth(rightColumnWidth), defender._mode);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Reconnector) {

            AlienGui::BeginIndent();

            // Restrict to color
            auto& reconnector = std::get<ReconnectorGenomeDescription>(node._cellTypeData);
            AlienGui::ComboOptionalColor(AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), reconnector._restrictToColor);

            // Restrict to mutants
            AlienGui::Combo(
                AlienGui::ComboParameters().name("Restrict to mutants").values(Const::ReconnectorRestrictToMutantStrings).textWidth(rightColumnWidth),
                reconnector._restrictToCreatures);

            AlienGui::EndIndent();

        } else if (nodeType == CellTypeGenome_Detonator) {

            AlienGui::BeginIndent();

            // Countdown
            auto& detonator = std::get<DetonatorGenomeDescription>(node._cellTypeData);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Countdown").textWidth(rightColumnWidth), detonator._countdown);

            AlienGui::EndIndent();
        }
    }
    ImGui::EndChild();
}

void _NodeEditorWidget::processNoSelection()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Selected node"));
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
    AlienGui::Group(AlienGui::GroupParameters().text("Neural network"));

    auto& node = _editData->getSelectedNodeRef();
    _neuralNetWidget->process(node._neuralNetwork._weights, node._neuralNetwork._biases, node._neuralNetwork._activationFunctions);
}
