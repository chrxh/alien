#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "EngineInterface/Colors.h"
#include "StyleRepository.h"

namespace
{
    auto const MaxContentTextWidth = 150.0f;
}

_GenomeEditorWindow ::_GenomeEditorWindow()
    : _AlienWindow("Genome editor", "windows.genome editor", false)
{
    _tabDatas = {TabData()};
}

_GenomeEditorWindow::~_GenomeEditorWindow()
{
}

namespace
{
    std::string generateShortDescription(int index, CellGenomeDescription const& cell)
    {
        return "No. " + std::to_string(index + 1) + ", Type: " + Const::CellFunctionToStringMap.at(cell.getCellFunctionType())
            + ", Color: " + std::to_string(cell.color);
    }
}

void _GenomeEditorWindow::processIntern()
{
    showToolbar();

    if (ImGui::BeginTabBar("##", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            _tabDatas.emplace_back(TabData());
        }
        AlienImGui::Tooltip("New gene");

        for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {
            if (ImGui::BeginTabItem(("Gene " + std::to_string(index + 1)).c_str(), NULL, ImGuiTabItemFlags_None)) {
                showGenomeTab(tabData);
                _currentTabIndex = toInt(index);
                ImGui::EndTabItem();
            }
        }

        ImGui::EndTabBar();
    }
}

void _GenomeEditorWindow::showToolbar()
{
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
        auto& tabData = _tabDatas.at(_currentTabIndex);
        tabData.genome.emplace_back(CellGenomeDescription());
    }
    AlienImGui::Tooltip("Add cell to gene description");
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
    }
    AlienImGui::Tooltip("Delete cell from gene description");
    AlienImGui::Separator();
}

void _GenomeEditorWindow::showGenomeTab(TabData& tabData)
{
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - _previewHeight), true)) {
        AlienImGui::Group("Genotype");
        showGenotype(tabData);
    }
    ImGui::EndChild();
    ImGui::Button("", ImVec2(-1, StyleRepository::getInstance().scaleContent(5.0f)));
    if (ImGui::IsItemActive()) {
        _previewHeight -= ImGui::GetIO().MouseDelta.y;
    }
    if (ImGui::BeginChild("##child3", ImVec2(0, 0), true)) {
        AlienImGui::Group("Phenotype");
        showPhenotype(tabData);
    }
    ImGui::EndChild();
}

namespace 
{
    class DynamicTableLayout
    {
    public:
        bool begin()
        {
            auto width = StyleRepository::getInstance().scaleContent(ImGui::GetContentRegionAvail().x);
            _columns = std::max(toInt(width / 260), 1);
            auto result = ImGui::BeginTable("##", _columns, ImGuiTableFlags_None);
            if (result) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
            }
            return result;
        }
        void end() { ImGui::EndTable(); }

        void next()
        {
            auto currentCol = (++_elementNumber) % _columns;
            if (currentCol > 0) {
                ImGui::TableSetColumnIndex(currentCol);
            } else {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
            }
        }

    private:
        int _columns = 0;
        int _elementNumber = 0;
    };
}

void _GenomeEditorWindow::showGenotype(TabData& tabData)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false)) {
        int index = 0;
        for (auto& cell : tabData.genome) {
            ImGui::PushID(index);

            float h, s, v;
            AlienImGui::convertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(h, s * 0.5f, v));
            ImGuiTreeNodeFlags flags = /*ImGuiTreeNodeFlags_Framed | */ ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow;
            if (tabData.selected && *tabData.selected == index) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            auto treeNodeOpen = ImGui::TreeNodeEx((generateShortDescription(index, cell) + "###").c_str(), flags);
            ImGui::PopStyleColor();
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                if (tabData.selected && *tabData.selected == index) {
                    tabData.selected.reset();
                } else {
                    tabData.selected = index;
                }
            }
            if (treeNodeOpen) {
                auto type = cell.getCellFunctionType();

                //cell type
                DynamicTableLayout table;
                if (table.begin()) {
                    if (AlienImGui::Combo(
                            AlienImGui::ComboParameters().name("Specialization").values(Const::CellFunctionStrings).textWidth(MaxContentTextWidth), type)) {
                        switch (type) {
                        case Enums::CellFunction_Neuron: {
                            cell.cellFunction = NeuronGenomeDescription();
                        } break;
                        case Enums::CellFunction_Transmitter: {
                            cell.cellFunction = TransmitterGenomeDescription();
                        } break;
                        case Enums::CellFunction_Constructor: {
                            cell.cellFunction = ConstructorGenomeDescription();
                        } break;
                        case Enums::CellFunction_Sensor: {
                            cell.cellFunction = SensorGenomeDescription();
                        } break;
                        case Enums::CellFunction_Nerve: {
                            cell.cellFunction = NerveGenomeDescription();
                        } break;
                        case Enums::CellFunction_Attacker: {
                            cell.cellFunction = AttackerGenomeDescription();
                        } break;
                        case Enums::CellFunction_Injector: {
                            cell.cellFunction = InjectorGenomeDescription();
                        } break;
                        case Enums::CellFunction_Muscle: {
                            cell.cellFunction = MuscleGenomeDescription();
                        } break;
                        case Enums::CellFunction_Placeholder1: {
                            cell.cellFunction = PlaceHolderGenomeDescription1();
                        } break;
                        case Enums::CellFunction_Placeholder2: {
                            cell.cellFunction = PlaceHolderGenomeDescription2();
                        } break;
                        case Enums::CellFunction_None: {
                            cell.cellFunction.reset();
                        } break;
                        }
                    }
                    table.next();

                    AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(MaxContentTextWidth), cell.color);
                    table.next();
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Distance").textWidth(MaxContentTextWidth).format("%.2f"), cell.referenceDistance);
                    table.next();
                    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Angle").textWidth(MaxContentTextWidth).format("%.1f"), cell.referenceAngle);
                    table.next();
                    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Max connections").textWidth(MaxContentTextWidth), cell.maxConnections);
                    table.next();
                    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Execution order").textWidth(MaxContentTextWidth), cell.executionOrderNumber);
                    table.next();
                    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block input").textWidth(MaxContentTextWidth), cell.inputBlocked);
                    table.next();
                    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block output").textWidth(MaxContentTextWidth), cell.outputBlocked);

                    table.end();
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
            ++index;
        }
    }
    ImGui::EndChild();
}

void _GenomeEditorWindow::showPhenotype(TabData& tabData)
{
}


