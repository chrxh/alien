#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"


#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/PreviewDescriptionConverter.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "EditorModel.h"
#include "StyleRepository.h"

namespace
{
    auto const ContentTextWidth = 165.0f;
    auto const WeightsAndBiasTextWidth = 100.0f;
    auto const WeightsAndBiasSelectionTextWidth = 400.0f;
    auto const DynamicTableColumnWidth = 240.0f;
}

_GenomeEditorWindow ::_GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController)
    : _AlienWindow("Genome editor", "windows.genome editor", false)
    , _editorModel(editorModel)
    , _simulationController(simulationController)
{
    _tabDatas = {TabData()};
}

_GenomeEditorWindow::~_GenomeEditorWindow()
{
}

void _GenomeEditorWindow::openTab(GenomeDescription const& genome)
{
    setOn(true);
    std::optional<int> tabIndex;
    for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {
        if (tabData.genome == genome) {
            tabIndex = toInt(index);
        }
    }
    if (tabIndex) {
        _tabIndexToSelect = *tabIndex;
    } else {
        TabData tabData;
        tabData.genome = genome;
        _tabToAdd = tabData;
    }
}

namespace
{
    std::string generateShortDescription(int index, CellGenomeDescription const& cell)
    {
        return "No. " + std::to_string(index + 1) + ", Type: " + Const::CellFunctionToStringMap.at(cell.getCellFunctionType())
            + ", Color: " + std::to_string(cell.color) + ", Angle: " + StringHelper::format(cell.referenceAngle, 1)
            + ", Distance: " + StringHelper::format(cell.referenceDistance, 2);
    }
}

void _GenomeEditorWindow::processIntern()
{
    processToolbar();

    if (ImGui::BeginTabBar("##", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            _tabDatas.emplace_back(TabData());
        }
        AlienImGui::Tooltip("New genome");

        std::optional<int> tabIndexToSelect = _tabIndexToSelect;
        std::optional<int> tabToDelete;
        _tabIndexToSelect.reset();

        //process tabs
        for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {

            bool open = true;
            bool* openPtr = nullptr;
            if (_tabDatas.size() > 1) {
                openPtr = &open;
            }
            int flags = (tabIndexToSelect && *tabIndexToSelect == index) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            if (ImGui::BeginTabItem(("Genome " + std::to_string(index + 1)).c_str(), openPtr, flags)) {
                _selectedTabIndex = toInt(index);
                processTab(tabData);
                ImGui::EndTabItem();
            }
            if (openPtr && *openPtr == false) {
                tabToDelete = toInt(index);
            }
        }

        //modify tabs
        if (tabToDelete.has_value()) {
            _tabDatas.erase(_tabDatas.begin() + *tabToDelete);
            if (_selectedTabIndex == _tabDatas.size()) {
                _selectedTabIndex = toInt(_tabDatas.size() - 1);
            }
        }
        if (_tabToAdd.has_value()) {
            if (_tabDatas.size() == 1 && _tabDatas.front().genome.empty()) {
                _tabDatas.clear();
            }
            _tabDatas.emplace_back(*_tabToAdd);
            _tabToAdd.reset();
        }

        ImGui::EndTabBar();
    }
}

void _GenomeEditorWindow::processToolbar()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
        CellGenomeDescription newNode;
        if (tabData.selectedNode) {
            tabData.genome.insert(tabData.genome.begin() + *tabData.selectedNode + 1, newNode);
            ++(*tabData.selectedNode);
        } else {
            tabData.genome.emplace_back(newNode);
            tabData.selectedNode = toInt(tabData.genome.size() - 1);
        }
    }
    AlienImGui::Tooltip("Add node");

    ImGui::SameLine();
    ImGui::BeginDisabled(tabData.genome.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
        if (tabData.selectedNode) {
            tabData.genome.erase(tabData.genome.begin() + *tabData.selectedNode);
            if (*tabData.selectedNode == toInt(tabData.genome.size())) {
                if (--(*tabData.selectedNode) < 0) {
                    tabData.selectedNode.reset();
                }
            }
        } else {
            tabData.genome.pop_back();
            if (!tabData.genome.empty()) {
                tabData.selectedNode = toInt(tabData.genome.size() - 1);
            }
        }
        _collapseAllNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete node");

    ImGui::SameLine();
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selected = selectedTab.selectedNode;
    ImGui::BeginDisabled(!(selected && *selected > 0));
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_UP)) {
        std::swap(selectedTab.genome.at(*selected), selectedTab.genome.at(*selected - 1));
        --(*selected);
        _collapseAllNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Decrease sequence number of selected node");

    ImGui::SameLine();
    ImGui::BeginDisabled(!(selected && *selected < selectedTab.genome.size() - 1));
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_DOWN)) {
        std::swap(selectedTab.genome.at(*selected), selectedTab.genome.at(*selected + 1));
        ++(*selected);
        _collapseAllNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Increase sequence number of selected node");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _editorModel->setCopiedGenome(GenomeDescriptionConverter::convertDescriptionToBytes(selectedTab.genome));
    }
    AlienImGui::Tooltip("Copy genome");

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
        AlienImGui::Group("Construction sequence");
        processGenomeEditTab(tab);
    }
    ImGui::EndChild();
    ImGui::Button("", ImVec2(-1, StyleRepository::getInstance().contentScale(5.0f)));
    if (ImGui::IsItemActive()) {
        _previewHeight -= ImGui::GetIO().MouseDelta.y;
    }
    AlienImGui::Group("Preview (approximation)");
    if (ImGui::BeginChild("##child4", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        showPreview(tab);
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
            auto width = StyleRepository::getInstance().contentScale(ImGui::GetContentRegionAvail().x);
            _columns = std::max(toInt(width / DynamicTableColumnWidth), 1);
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

    void applyNewCellFunction(CellGenomeDescription&cell, CellFunction type)
    {
        switch (type) {
        case CellFunction_Neuron: {
            cell.cellFunction = NeuronGenomeDescription();
        } break;
        case CellFunction_Transmitter: {
            cell.cellFunction = TransmitterGenomeDescription();
        } break;
        case CellFunction_Constructor: {
            cell.cellFunction = ConstructorGenomeDescription();
        } break;
        case CellFunction_Sensor: {
            cell.cellFunction = SensorGenomeDescription();
        } break;
        case CellFunction_Nerve: {
            cell.cellFunction = NerveGenomeDescription();
        } break;
        case CellFunction_Attacker: {
            cell.cellFunction = AttackerGenomeDescription();
        } break;
        case CellFunction_Injector: {
            cell.cellFunction = InjectorGenomeDescription();
        } break;
        case CellFunction_Muscle: {
            cell.cellFunction = MuscleGenomeDescription();
        } break;
        case CellFunction_Placeholder1: {
            cell.cellFunction = PlaceHolderGenomeDescription1();
        } break;
        case CellFunction_Placeholder2: {
            cell.cellFunction = PlaceHolderGenomeDescription2();
        } break;
        case CellFunction_None: {
            cell.cellFunction.reset();
        } break;
        }
    }
}

void _GenomeEditorWindow::processGenomeEditTab(TabData& tab)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false)) {
        int index = 0;
        for (auto& cell : tab.genome) {
            ImGui::PushID(index);

            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(h, s * 0.5f, v));
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow;
            if (tab.selectedNode && *tab.selectedNode == index) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            if (_nodeIndexToJump && *_nodeIndexToJump == index) {
                ImGui::SetScrollHereY();
                _nodeIndexToJump = std::nullopt;
            }

            if (_collapseAllNodes) {
                ImGui::SetNextTreeNodeOpen(false);
            }
            auto treeNodeOpen = ImGui::TreeNodeEx((generateShortDescription(index, cell) + "###").c_str(), flags);
            ImGui::PopStyleColor();
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                if (tab.selectedNode && *tab.selectedNode == index) {
                    tab.selectedNode.reset();
                } else {
                    tab.selectedNode = index;
                }
            }
            if (ImGui::IsItemToggledOpen()) {
                tab.selectedNode = index;
            }

            if (treeNodeOpen) {
                auto origCell = cell;
                processNodeEdit(tab, cell);
                if (origCell != cell) {
                    tab.selectedNode = index;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
            ++index;
        }
    }
    ImGui::EndChild();
    _collapseAllNodes = false;
}

void _GenomeEditorWindow::processNodeEdit(TabData& tab, CellGenomeDescription& cell)
{
    auto type = cell.getCellFunctionType();

    //cell type
    DynamicTableLayout table;
    if (table.begin()) {
        if (AlienImGui::CellFunctionCombo(AlienImGui::CellFunctionComboParameters().name("Specialization").textWidth(ContentTextWidth), type)) {
            applyNewCellFunction(cell, type);
        }
        table.next();

        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(ContentTextWidth), cell.color);
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Distance").textWidth(ContentTextWidth).format("%.2f").step(0.1f), cell.referenceDistance);
        table.next();
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Angle").textWidth(ContentTextWidth).format("%.1f"), cell.referenceAngle);
        table.next();
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Max connections").textWidth(ContentTextWidth), cell.maxConnections);
        table.next();
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Execution order").textWidth(ContentTextWidth), cell.executionOrderNumber);
        table.next();
        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block input").textWidth(ContentTextWidth), cell.inputBlocked);
        table.next();
        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block output").textWidth(ContentTextWidth), cell.outputBlocked);

        switch (type) {
        case CellFunction_Neuron: {
        } break;
        case CellFunction_Transmitter: {
            auto& transmitter = std::get<TransmitterGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Energy distribution")
                    .values({"Connected cells", "Transmitters and Constructors"})
                    .textWidth(ContentTextWidth),
                transmitter.mode);
        } break;
        case CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Single construction").textWidth(ContentTextWidth), constructor.singleConstruction);
            table.next();
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters().name("Separate construction").textWidth(ContentTextWidth), constructor.separateConstruction);
            table.next();
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Adapt max connections").textWidth(ContentTextWidth), constructor.adaptMaxConnections);
            int constructorMode = constructor.mode == 0 ? 0 : 1;
            table.next();
            if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Activation mode").textWidth(ContentTextWidth).values({"Manual", "Automatic"}), constructorMode)) {
                constructor.mode = constructorMode;
            }
            if (constructorMode == 1) {
                table.next();
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Periodicity").textWidth(ContentTextWidth), constructor.mode);
                if (constructor.mode < 0) {
                    constructor.mode = 0;
                }
            }
            table.next();
            AlienImGui::AngleAlignmentCombo(
                AlienImGui::AngleAlignmentComboParameters().name("Angle alignment").textWidth(ContentTextWidth), constructor.angleAlignment);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Offspring stiffness").format("%.2f").step(0.05f).textWidth(ContentTextWidth),
                constructor.stiffness);
            table.next();
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Offspring activation time").textWidth(ContentTextWidth), constructor.constructionActivationTime);
        } break;
        case CellFunction_Sensor: {
            auto& sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
            auto sensorMode = sensor.getSensorMode();

            table.next();
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters().name("Mode").textWidth(ContentTextWidth).values({"Scan vicinity", "Scan specific direction"}), sensorMode)) {
                if (sensorMode == SensorMode_Neighborhood) {
                    sensor.fixedAngle.reset();
                } else {
                    sensor.fixedAngle = 0.0f;
                }
            }
            if (sensorMode == SensorMode_FixedAngle) {
                table.next();
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Scan angle").textWidth(ContentTextWidth).format("%.1f"), *sensor.fixedAngle);
            }
            table.next();
            AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Scan color").textWidth(ContentTextWidth), sensor.color);

            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Min density").format("%.2f").step(0.05f).textWidth(ContentTextWidth), sensor.minDensity);
        } break;
        case CellFunction_Nerve: {
            auto& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
            bool pulseGeneration = nerve.pulseMode > 0;
            table.next();
            if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Pulse generation").textWidth(ContentTextWidth), pulseGeneration)) {
                nerve.pulseMode = pulseGeneration ? 1 : 0;
            }
            if (pulseGeneration) {
                table.next();
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Periodicity").textWidth(ContentTextWidth), nerve.pulseMode);
                bool alternation = nerve.alternationMode > 0;
                table.next();
                if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(ContentTextWidth), alternation)) {
                    nerve.alternationMode = alternation ? 1 : 0;
                }
                if (alternation) {
                    table.next();
                    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Number of pulses").textWidth(ContentTextWidth), nerve.alternationMode);
                }
            }
        } break;
        case CellFunction_Attacker: {
            auto& attacker = std::get<AttackerGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Energy distribution")
                    .values({"Connected cells", "Transmitters and Constructors"})
                    .textWidth(ContentTextWidth),
                attacker.mode);
        } break;
        case CellFunction_Injector: {
        } break;
        case CellFunction_Muscle: {
            auto& muscle = std::get<MuscleGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Mode").values({"Movement", "Contraction and expansion", "Bending"}).textWidth(ContentTextWidth),
                muscle.mode);
        } break;
        case CellFunction_Placeholder1: {
        } break;
        case CellFunction_Placeholder2: {
        } break;
        case CellFunction_None: {
        } break;
        }

        table.end();

        switch (type) {
        case CellFunction_Neuron: {
            auto& neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
            if (ImGui::TreeNodeEx("Neural network", ImGuiTreeNodeFlags_None)) {
                AlienImGui::NeuronSelection(
                    AlienImGui::NeuronSelectionParameters().outputButtonPositionFromRight(WeightsAndBiasSelectionTextWidth),
                    neuron.weights,
                    neuron.bias,
                    _selectedInput,
                    _selectedOutput);
                DynamicTableLayout table;
                if (table.begin()) {
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(WeightsAndBiasTextWidth),
                        neuron.weights[_selectedOutput][_selectedInput]);
                    table.next();
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(WeightsAndBiasTextWidth), neuron.bias[_selectedOutput]);
                    table.end();
                }
                ImGui::TreePop();
            }
        } break;
        case CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
            std::string content;
            if (constructor.isMakeGenomeCopy()) {
                content = "Make copy of this genome";
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
                _editorModel->setCopiedGenome(constructor.isMakeGenomeCopy() ? GenomeDescriptionConverter::convertDescriptionToBytes(tab.genome) : constructor.getGenomeData());
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(!_editorModel->getCopiedGenome().has_value());
            if (AlienImGui::Button("Paste")) {
                constructor.genome = *_editorModel->getCopiedGenome();
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (AlienImGui::Button("This")) {
                constructor.setMakeGenomeCopy();
            }
            ImGui::SameLine();
            if (AlienImGui::Button("Edit")) {
                auto genomeToOpen = constructor.isMakeGenomeCopy() ? tab.genome : GenomeDescriptionConverter::convertBytesToDescription(constructor.getGenomeData(), _simulationController->getSimulationParameters());
                openTab(genomeToOpen);
            }
        } break;
        }
    }
    validationAndCorrection(cell);
}

void _GenomeEditorWindow::showPreview(TabData& tab)
{
    auto const& genome = _tabDatas.at(_selectedTabIndex).genome;
    auto preview = PreviewDescriptionConverter::convert(genome, tab.selectedNode, _simulationController->getSimulationParameters());
    if (AlienImGui::ShowPreviewDescription(preview, _genomeZoom, tab.selectedNode)) {
        _nodeIndexToJump = tab.selectedNode;
    }
}

void _GenomeEditorWindow::validationAndCorrection(CellGenomeDescription& cell) const
{
    auto numExecutionOrderNumbers = _simulationController->getSimulationParameters().cellMaxExecutionOrderNumbers;
    auto maxBonds = _simulationController->getSimulationParameters().cellMaxBonds;
    cell.color = (cell.color + MAX_COLORS) % MAX_COLORS;
    cell.executionOrderNumber = (cell.executionOrderNumber + numExecutionOrderNumbers) % numExecutionOrderNumbers;
    cell.maxConnections = (cell.maxConnections + maxBonds + 1) % (maxBonds + 1);
    cell.referenceDistance = std::max(0.0f, cell.referenceDistance);

    switch (cell.getCellFunctionType()) {
    case CellFunction_Constructor: {
        auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
        if (constructor.mode < 0) {
            constructor.mode = 0;
        }
        constructor.stiffness = std::max(0.0f, std::min(1.0f, constructor.stiffness));
    } break;
    case CellFunction_Sensor: {
        auto& sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
        sensor.minDensity = std::max(0.0f, std::min(1.0f, sensor.minDensity));
    } break;
    case CellFunction_Nerve: {
        auto& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
        nerve.pulseMode = std::max(0, nerve.pulseMode);
        nerve.alternationMode = std::max(0, nerve.alternationMode);
    } break;
    }
}

