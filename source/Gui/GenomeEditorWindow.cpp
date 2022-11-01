#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/GenomeTranslator.h"
#include "EngineInterface/Colors.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "StyleRepository.h"

namespace
{
    auto const MaxContentTextWidth = 150.0f;
    auto const MaxContentWidth = 240.0f;
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
    processToolbar();

    if (ImGui::BeginTabBar("##", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            _tabDatas.emplace_back(TabData());
        }
        AlienImGui::Tooltip("New gene");

        std::optional<int> tabIndexToSelect = _tabIndexToSelect;
        std::optional<int> tabToDelete;
        _tabIndexToSelect.reset();

        for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {

            bool open = true;
            bool* openPtr = nullptr;
            if (_tabDatas.size() > 1) {
                openPtr = &open;
            }
            int flags = (tabIndexToSelect && *tabIndexToSelect == index) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            if (ImGui::BeginTabItem(("Gene " + std::to_string(index + 1)).c_str(), openPtr, flags)) {
                processTab(tabData);
                _selectedTabIndex = toInt(index);
                ImGui::EndTabItem();
            }
            if (openPtr && *openPtr == false) {
                tabToDelete = toInt(index);
            }
        }

        if (tabToDelete.has_value()) {
            _tabDatas.erase(_tabDatas.begin() + *tabToDelete);
            if (_selectedTabIndex == _tabDatas.size()) {
                _selectedTabIndex = toInt(_tabDatas.size() - 1);
            }
        }

        ImGui::EndTabBar();
    }
}

void _GenomeEditorWindow::processToolbar()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
        if (tabData.selected) {
            tabData.genome.insert(tabData.genome.begin() + *tabData.selected + 1, CellGenomeDescription());
            ++(*tabData.selected);
        } else {
            tabData.genome.emplace_back(CellGenomeDescription());
            tabData.selected = toInt(tabData.genome.size() - 1);
        }
    }
    AlienImGui::Tooltip("Add cell node to gene description");

    ImGui::SameLine();
    ImGui::BeginDisabled(tabData.genome.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
        if (tabData.selected) {
            tabData.genome.erase(tabData.genome.begin() + *tabData.selected);
            if (*tabData.selected == toInt(tabData.genome.size())) {
                if (--(*tabData.selected) < 0) {
                    tabData.selected.reset();
                }
            }
        } else {
            tabData.genome.pop_back();
            if (!tabData.genome.empty()) {
                tabData.selected = toInt(tabData.genome.size() - 1);
            }
        }
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete cell node from gene description");

    ImGui::SameLine();
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selected = selectedTab.selected;
    ImGui::BeginDisabled(!(selected && *selected > 0));
    if (AlienImGui::ToolbarButton(ICON_FA_ARROW_UP)) {
        std::swap(selectedTab.genome.at(*selected), selectedTab.genome.at(*selected - 1));
        --(*selected);
        _collapseAllNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Decrease sequence number of selected cell node");

    ImGui::SameLine();
    ImGui::BeginDisabled(!(selected && *selected < selectedTab.genome.size() - 1));
    if (AlienImGui::ToolbarButton(ICON_FA_ARROW_DOWN)) {
        std::swap(selectedTab.genome.at(*selected), selectedTab.genome.at(*selected + 1));
        ++(*selected);
        _collapseAllNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Increase sequence number of selected cell node");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedGenome = GenomeTranslator::encode(selectedTab.genome);
    }
    AlienImGui::Tooltip("Copy selected node");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS_SQUARE)) {
        _collapseAllNodes = true;
    }
    AlienImGui::Tooltip("Collapse all nodes");

    AlienImGui::Separator();
}

void _GenomeEditorWindow::processTab(TabData& tab)
{
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - _previewHeight), true)) {
        AlienImGui::Group("Genotype");
        processGenotype(tab);
    }
    ImGui::EndChild();
    ImGui::Button("", ImVec2(-1, StyleRepository::getInstance().scaleContent(5.0f)));
    if (ImGui::IsItemActive()) {
        _previewHeight -= ImGui::GetIO().MouseDelta.y;
    }
    if (ImGui::BeginChild("##child3", ImVec2(0, 0), true)) {
        AlienImGui::Group("Phenotype");
        showPhenotype(tab);
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
            _columns = std::max(toInt(width / MaxContentWidth), 1);
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

    void applyNewCellFunction(CellGenomeDescription&cell, Enums::CellFunction type)
    {
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
}

void _GenomeEditorWindow::processGenotype(TabData& tab)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false)) {
        int index = 0;
        for (auto& cell : tab.genome) {
            ImGui::PushID(index);

            float h, s, v;
            AlienImGui::convertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(h, s * 0.5f, v));
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow;
            if (tab.selected && *tab.selected == index) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }

            if (_collapseAllNodes) {
                ImGui::SetNextTreeNodeOpen(false);
            }
            auto treeNodeOpen = ImGui::TreeNodeEx((generateShortDescription(index, cell) + "###").c_str(), flags);
            ImGui::PopStyleColor();
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                if (tab.selected && *tab.selected == index) {
                    tab.selected.reset();
                } else {
                    tab.selected = index;
                }
            }

            if (treeNodeOpen) {
                processCell(tab, cell);
                ImGui::TreePop();
            }
            ImGui::PopID();
            ++index;
        }
    }
    ImGui::EndChild();
    _collapseAllNodes = false;
}

void _GenomeEditorWindow::processCell(TabData& tab, CellGenomeDescription& cell)
{
    auto type = cell.getCellFunctionType();

    //cell type
    DynamicTableLayout table;
    if (table.begin()) {
        if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Specialization").values(Const::CellFunctionStrings).textWidth(MaxContentTextWidth), type)) {
            applyNewCellFunction(cell, type);
        }
        table.next();

        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(MaxContentTextWidth), cell.color);
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Distance").textWidth(MaxContentTextWidth).format("%.2f").step(0.1f), cell.referenceDistance);
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

        switch (type) {
        case Enums::CellFunction_Neuron: {
        } break;
        case Enums::CellFunction_Transmitter: {
        } break;
        case Enums::CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Single construction").textWidth(MaxContentTextWidth), constructor.singleConstruction);
            table.next();
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters().name("Separate construction").textWidth(MaxContentTextWidth), constructor.separateConstruction);
            table.next();
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Make sticky").textWidth(MaxContentTextWidth), constructor.makeSticky);
            int constructorMode = constructor.mode == 0 ? 0 : 1;
            table.next();
            if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Mode").textWidth(MaxContentTextWidth).values({"Manual", "Automatic"}), constructorMode)) {
                constructor.mode = constructorMode;
            }
            if (constructorMode == 1) {
                table.next();
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Cycles").textWidth(MaxContentTextWidth), constructor.mode);
                if (constructor.mode < 0) {
                    constructor.mode = 0;
                }
            }
        } break;
        case Enums::CellFunction_Sensor: {
            auto& sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
            auto sensorMode = sensor.getSensorMode();

            table.next();
            AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Scan color").textWidth(MaxContentTextWidth), sensor.color);

            table.next();
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters().name("Mode").textWidth(MaxContentTextWidth).values({"Neighborhood", "Fixed angle scan"}), sensorMode)) {
                if (sensorMode == Enums::SensorMode_Neighborhood) {
                    sensor.fixedAngle.reset();
                } else {
                    sensor.fixedAngle = 0.0f;
                }
            }
            if (sensorMode == Enums::SensorMode_FixedAngle) {
                table.next();
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Scan angle").textWidth(MaxContentTextWidth).format("%.1f"), *sensor.fixedAngle);
            }
        } break;
        case Enums::CellFunction_Nerve: {
        } break;
        case Enums::CellFunction_Attacker: {
        } break;
        case Enums::CellFunction_Injector: {
        } break;
        case Enums::CellFunction_Muscle: {
        } break;
        case Enums::CellFunction_Placeholder1: {
        } break;
        case Enums::CellFunction_Placeholder2: {
        } break;
        case Enums::CellFunction_None: {
        } break;
        }

        table.end();

        switch (type) {
        case Enums::CellFunction_Neuron: {
            auto& neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
            if (ImGui::TreeNodeEx("Weight matrix", ImGuiTreeNodeFlags_None)) {
                AlienImGui::InputFloatMatrix(AlienImGui::InputFloatMatrixParameters().step(0.1f), neuron.weights);
                ImGui::TreePop();
            }
            if (ImGui::TreeNodeEx("Bias", ImGuiTreeNodeFlags_None)) {
                AlienImGui::InputFloatVector(AlienImGui::InputFloatVectorParameters().step(0.1f), neuron.bias);
                ImGui::TreePop();
            }
        } break;
        case Enums::CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
            std::string content;
            if (constructor.isMakeGenomeCopy()) {
                content = "Make copy of this gene";
            } else {
                auto size = constructor.getGenomeData().size();
                if (size > 0) {
                    content = std::to_string(size) + " bytes of genetic information";
                } else {
                    content = "No genetic information";
                }
            }
            auto width = ImGui::GetContentRegionAvail().x / 2;
            if (ImGui::BeginChild("##", ImVec2(width, ImGui::GetTextLineHeight() * 2 /*+ ImGui::GetStyle().FramePadding.y*2*/), true)) {
                AlienImGui::MonospaceText(content);
            }
            ImGui::EndChild();
            if (AlienImGui::Button("Clear")) {
                constructor.setGenome({});
            }
            ImGui::SameLine();
            if (AlienImGui::Button("Copy")) {
                _copiedGenome = constructor.isMakeGenomeCopy() ? GenomeTranslator::encode(tab.genome) : constructor.getGenomeData();
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(!_copiedGenome.has_value());
            if (AlienImGui::Button("Paste")) {
                constructor.genome = *_copiedGenome;
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (AlienImGui::Button("This")) {
                constructor.setMakeGenomeCopy();
            }
            ImGui::SameLine();
            if (AlienImGui::Button("View")) {
                auto copiedGenome = constructor.isMakeGenomeCopy() ? GenomeTranslator::encode(tab.genome) : constructor.getGenomeData();
                if (auto index = findTabToGenomeData(copiedGenome)) {
                    _tabIndexToSelect = *index;
                } else {
                    TabData tabData;
                    tabData.genome = GenomeTranslator::decode(copiedGenome);
                    _tabDatas.emplace_back(tabData);
                }
            }
        } break;
        }
    }
}

void _GenomeEditorWindow::showPhenotype(TabData& tab)
{

}

std::optional<int> _GenomeEditorWindow::findTabToGenomeData(std::vector<uint8_t> const& genome) const
{
    for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {
        auto genomeFromTab = GenomeTranslator::encode(tabData.genome);
        if (genomeFromTab == genome) {
            return toInt(index);
        }
    }
    return std::nullopt;
}


