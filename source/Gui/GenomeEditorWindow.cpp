#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "EngineInterface/Colors.h"

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
    std::string generateShortDescription(CellGenomeDescription const& cell)
    {
        return "Type: " + Const::CellFunctionToStringMap.at(cell.getCellFunctionType()) + ", Color: " + std::to_string(cell.color);
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
            if (ImGui::BeginTabItem(("Gene " + std::to_string(index)).c_str(), NULL, ImGuiTabItemFlags_None)) {
                showGenomeContent(tabData);
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

void _GenomeEditorWindow::showGenomeContent(TabData& tabData)
{
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - _previewHeight), true)) {
        AlienImGui::Group("Genotype");
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
            auto treeNodeOpen = ImGui::TreeNodeEx((generateShortDescription(cell) + "###").c_str(), flags);
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

                //color
                ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - MaxContentTextWidth);
                AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(MaxContentTextWidth), cell.color);
                ImGui::PopItemWidth();
                ImGui::SameLine();
                AlienImGui::Text("Color");

                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Distance").textWidth(MaxContentTextWidth), cell.referenceDistance);
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Angle").textWidth(MaxContentTextWidth), cell.referenceAngle);
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Max connections").textWidth(MaxContentTextWidth), cell.maxConnections);
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Execution order").textWidth(MaxContentTextWidth), cell.maxConnections);
                AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block input").textWidth(MaxContentTextWidth), cell.inputBlocked);
                AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block output").textWidth(MaxContentTextWidth), cell.outputBlocked);

                ImGui::TreePop();
            }
            ImGui::PopID();
            ++index;
        }
    }
    ImGui::EndChild();
    ImGui::InvisibleButton("hsplitter", ImVec2(-1, 8.0f));
    if (ImGui::IsItemActive()) {
        _previewHeight -= ImGui::GetIO().MouseDelta.y;
    }
    ImGui::BeginChild("##child3", ImVec2(0, 0), true);
    AlienImGui::Group("Phenotype");
    ImGui::EndChild();
}


