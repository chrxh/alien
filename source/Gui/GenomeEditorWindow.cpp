#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <ImFileDialog.h>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"


#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/PreviewDescriptionConverter.h"
#include "EngineInterface/Serializer.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "EditorModel.h"
#include "GenericFileDialogs.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto const ContentTextWidth = 165.0f;
    auto const WeightsAndBiasTextWidth = 100.0f;
    auto const WeightsAndBiasSelectionTextWidth = 400.0f;
    auto const DynamicTableColumnWidth = 265.0f;
}

_GenomeEditorWindow ::_GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController, Viewport const& viewport)
    : _AlienWindow("Genome editor", "windows.genome editor", false)
    , _editorModel(editorModel)
    , _simController(simulationController)
    , _viewport(viewport)
{
    _tabDatas = {TabData()};

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("windows.genome editor.starting path", path.string());
}

_GenomeEditorWindow::~_GenomeEditorWindow()
{
    GlobalSettings::getInstance().setStringState("windows.genome editor.starting path", _startingPath);
}

void _GenomeEditorWindow::openTab(GenomeDescription const& genome)
{
    setOn(true);
    if (_tabDatas.size() == 1 && _tabDatas.front().genome.empty()) {
        _tabDatas.clear();
    }
    std::optional<int> tabIndex;
    for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {
        if (tabData.genome == genome) {
            tabIndex = toInt(index);
        }
    }
    if (tabIndex) {
        _tabIndexToSelect = *tabIndex;
    } else {
        scheduleAddTab(genome);
    }
}

GenomeDescription const& _GenomeEditorWindow::getCurrentGenome() const
{
    return _tabDatas.at(_selectedTabIndex).genome;
}

namespace
{
    std::string generateShortDescription(int index, CellGenomeDescription const& cell)
    {
        return "No. " + std::to_string(index + 1) + ", Type: " + Const::CellFunctionToStringMap.at(cell.getCellFunctionType())
            + ", Color: " + std::to_string(cell.color) + ", Angle: " + StringHelper::format(cell.referenceAngle, 1)
            + ", Energy: " + StringHelper::format(cell.energy, 1);
    }
}

void _GenomeEditorWindow::processIntern()
{
    processToolbar();

    if (ImGui::BeginTabBar("##", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            scheduleAddTab(GenomeDescription());
        }
        AlienImGui::Tooltip("New genome");

        std::optional<int> tabIndexToSelect = _tabIndexToSelect;
        std::optional<int> tabToDelete;
        _tabIndexToSelect.reset();

        //process tabs
        for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(0)) {

            ImGui::PushID(tabData.id);
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
            ImGui::PopID();
        }

        //delete tab
        if (tabToDelete.has_value()) {
            _tabDatas.erase(_tabDatas.begin() + *tabToDelete);
            if (_selectedTabIndex == _tabDatas.size()) {
                _selectedTabIndex = toInt(_tabDatas.size() - 1);
            }
        }

        //add tab
        if (_tabToAdd.has_value()) {
            _tabDatas.emplace_back(*_tabToAdd);
            _tabToAdd.reset();
        }

        ImGui::EndTabBar();
    }
}

void _GenomeEditorWindow::processToolbar()
{
    if (_tabDatas.empty()) {
        return;
    }
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);

    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
        onOpenGenome();
    }
    AlienImGui::Tooltip("Open genome from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        onSaveGenome();
    }
    AlienImGui::Tooltip("Save genome to file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedGenome = GenomeDescriptionConverter::convertDescriptionToBytes(selectedTab.genome);
    }
    AlienImGui::Tooltip("Copy genome");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
        onAddNode();
    }
    AlienImGui::Tooltip("Add node");

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
        onDeleteNode();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete node");

    ImGui::SameLine();
    auto& selectedNode = selectedTab.selectedNode;
    ImGui::BeginDisabled(!(selectedNode && *selectedNode > 0));
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_UP)) {
        onNodeDecreaseSequenceNumber();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Decrease sequence number of selected node");

    ImGui::SameLine();
    ImGui::BeginDisabled(!(selectedNode && *selectedNode < selectedTab.genome.size() - 1));
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_DOWN)) {
        onNodeIncreaseSequenceNumber();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Increase sequence number of selected node");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS_SQUARE)) {
        _collapseAllNodes = true;
    }
    AlienImGui::Tooltip("Collapse all nodes");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SEEDLING)) {
        onCreateSpore();
    }
    AlienImGui::Tooltip("Create a spore with current genome");

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
        case CellFunction_Defender: {
            cell.cellFunction = DefenderGenomeDescription();
        } break;
        case CellFunction_Placeholder: {
            cell.cellFunction = PlaceHolderGenomeDescription();
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
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Angle").textWidth(ContentTextWidth).format("%.1f"), cell.referenceAngle);
        table.next();
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").textWidth(ContentTextWidth).format("%.1f"), cell.energy);
        table.next();
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Execution number").textWidth(ContentTextWidth), cell.executionOrderNumber);
        table.next();
        AlienImGui::InputOptionalInt(AlienImGui::InputIntParameters().name("Input execution number").textWidth(ContentTextWidth), cell.inputExecutionOrderNumber);
        table.next();
        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block output").textWidth(ContentTextWidth), cell.outputBlocked);
        table.next();
        AlienImGui::InputOptionalInt(
            AlienImGui::InputIntParameters().name("Required connections").textWidth(ContentTextWidth), cell.numRequiredAdditionalConnections);

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
            AlienImGui::InputOptionalInt(AlienImGui::InputIntParameters().name("Max connections").textWidth(ContentTextWidth), constructor.maxConnections);
            int constructorMode = constructor.mode == 0 ? 0 : 1;
            table.next();
            if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Activation mode").textWidth(ContentTextWidth).values({"Manual", "Automatic"}), constructorMode)) {
                constructor.mode = constructorMode;
            }
            if (constructorMode == 1) {
                table.next();
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Interval").textWidth(ContentTextWidth), constructor.mode);
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
            if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Generate pulses").textWidth(ContentTextWidth), pulseGeneration)) {
                nerve.pulseMode = pulseGeneration ? 1 : 0;
            }
            if (pulseGeneration) {
                table.next();
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Pulse interval").textWidth(ContentTextWidth), nerve.pulseMode);
                bool alternation = nerve.alternationMode > 0;
                table.next();
                if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(ContentTextWidth), alternation)) {
                    nerve.alternationMode = alternation ? 1 : 0;
                }
                if (alternation) {
                    table.next();
                    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Pulses per phase").textWidth(ContentTextWidth), nerve.alternationMode);
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
            auto& injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Mode").textWidth(ContentTextWidth).values({"Cells under construction", "All Cells"}), injector.mode);
        } break;
        case CellFunction_Muscle: {
            auto& muscle = std::get<MuscleGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Mode").values({"Movement", "Contraction and expansion", "Bending"}).textWidth(ContentTextWidth),
                muscle.mode);
        } break;
        case CellFunction_Defender: {
            auto& defender = std::get<DefenderGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Mode").values({"Anti-attacker", "Anti-injector"}).textWidth(ContentTextWidth),
                defender.mode);
        } break;
        case CellFunction_Placeholder: {
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
                    neuron.biases,
                    _selectedInput,
                    _selectedOutput);
                DynamicTableLayout table;
                if (table.begin()) {
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(WeightsAndBiasTextWidth),
                        neuron.weights[_selectedOutput][_selectedInput]);
                    table.next();
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(WeightsAndBiasTextWidth), neuron.biases[_selectedOutput]);
                    table.end();
                }
                ImGui::TreePop();
            }
        } break;
        case CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
            processSubGenomeWidgets(tab, constructor);
        } break;
        case CellFunction_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);
            processSubGenomeWidgets(tab, injector);
        } break;
        }
    }
    validationAndCorrection(cell);
}

template <typename Description>
void _GenomeEditorWindow::processSubGenomeWidgets(TabData const& tab, Description& desc)
{
    std::string content;
    if (desc.isMakeGenomeCopy()) {
        content = "Genome: self-copy";
    } else {
        auto size = desc.getGenomeData().size();
        if (size > 0) {
            content = "Genome: " + std::to_string(size) + " bytes";
        } else {
            content = "Genome: none";
        }
    }
    auto width = ImGui::GetContentRegionAvail().x / 2;
    if (ImGui::BeginChild("##", ImVec2(width, contentScale(60.0f)), true)) {
        AlienImGui::MonospaceText(content);
        if (AlienImGui::Button("Clear")) {
            desc.setGenome({});
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Copy")) {
            _copiedGenome = desc.isMakeGenomeCopy() ? GenomeDescriptionConverter::convertDescriptionToBytes(tab.genome) : desc.getGenomeData();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!_copiedGenome.has_value());
        if (AlienImGui::Button("Paste")) {
            desc.genome = *_copiedGenome;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (AlienImGui::Button("Edit")) {
            auto genomeToOpen = desc.isMakeGenomeCopy()
                ? tab.genome
                : GenomeDescriptionConverter::convertBytesToDescription(desc.getGenomeData());
            openTab(genomeToOpen);
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Set self-copy")) {
            desc.setMakeGenomeCopy();
        }
    }
    ImGui::EndChild();
}


void _GenomeEditorWindow::onOpenGenome()
{
    GenericFileDialogs::getInstance().showOpenFileDialog("Open genome", "Genome (*.genome){.genome},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        std::vector<uint8_t> genomeData;
        if (!Serializer::deserializeGenomeFromFile(genomeData, firstFilename.string())) {
            MessageDialog::getInstance().show("Open genome", "The selected file could not be opened.");
        } else {
            openTab(GenomeDescriptionConverter::convertBytesToDescription(genomeData));
        }
    });
}

void _GenomeEditorWindow::onSaveGenome()
{
    GenericFileDialogs::getInstance().showSaveFileDialog(
        "Save genome", "Genome (*.genome){.genome},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            auto const& selectedTab = _tabDatas.at(_selectedTabIndex);
            auto genomeData = GenomeDescriptionConverter::convertDescriptionToBytes(selectedTab.genome);
            if (!Serializer::serializeGenomeToFile(firstFilename.string(), genomeData)) {
                MessageDialog::getInstance().show("Save genome", "The selected file could not be saved.");
            }
        });
}

void _GenomeEditorWindow::onAddNode()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);

    CellGenomeDescription newNode;
    if (tabData.selectedNode) {
        newNode.color = tabData.genome.at(*tabData.selectedNode).color;
        tabData.genome.insert(tabData.genome.begin() + *tabData.selectedNode + 1, newNode);
        ++(*tabData.selectedNode);
    } else {
        if (!tabData.genome.empty()) {
            newNode.color = tabData.genome.back().color;
        }
        tabData.genome.emplace_back(newNode);
        tabData.selectedNode = toInt(tabData.genome.size() - 1);
    }
}

void _GenomeEditorWindow::onDeleteNode()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);
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

void _GenomeEditorWindow::onNodeDecreaseSequenceNumber()
{
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selectedNode = selectedTab.selectedNode;
    std::swap(selectedTab.genome.at(*selectedNode), selectedTab.genome.at(*selectedNode - 1));
    --(*selectedNode);
    _collapseAllNodes = true;
}

void _GenomeEditorWindow::onNodeIncreaseSequenceNumber()
{
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selectedNode = selectedTab.selectedNode;
    std::swap(selectedTab.genome.at(*selectedNode), selectedTab.genome.at(*selectedNode + 1));
    ++(*selectedNode);
    _collapseAllNodes = true;
}

void _GenomeEditorWindow::onCreateSpore()
{
    auto pos = _viewport->getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto genomeDesc = getCurrentGenome();
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);

    auto parameter = _simController->getSimulationParameters();
    auto cell = CellDescription()
                    .setPos(pos)
                    .setEnergy(parameter.cellNormalEnergy[_editorModel->getDefaultColorCode()] * (genomeDesc.size() * 2 + 1))
                    .setStiffness(1.0f)
                    .setMaxConnections(6)
                    .setExecutionOrderNumber(0)
                    .setColor(_editorModel->getDefaultColorCode())
                    .setCellFunction(ConstructorDescription().setGenome(genome));
    auto data = DataDescription().addCell(cell);
    _simController->addAndSelectSimulationData(data);
}

void _GenomeEditorWindow::showPreview(TabData& tab)
{
    auto const& genome = _tabDatas.at(_selectedTabIndex).genome;
    auto preview = PreviewDescriptionConverter::convert(genome, tab.selectedNode, _simController->getSimulationParameters());
    if (AlienImGui::ShowPreviewDescription(preview, _genomeZoom, tab.selectedNode)) {
        _nodeIndexToJump = tab.selectedNode;
    }
}

void _GenomeEditorWindow::validationAndCorrection(CellGenomeDescription& cell) const
{
    auto numExecutionOrderNumbers = _simController->getSimulationParameters().cellNumExecutionOrderNumbers;
    cell.color = (cell.color + MAX_COLORS) % MAX_COLORS;
    cell.executionOrderNumber = (cell.executionOrderNumber + numExecutionOrderNumbers) % numExecutionOrderNumbers;
    if (cell.inputExecutionOrderNumber) {
        cell.inputExecutionOrderNumber = (*cell.inputExecutionOrderNumber + numExecutionOrderNumbers) % numExecutionOrderNumbers;
    }
    if (cell.numRequiredAdditionalConnections) {
        cell.numRequiredAdditionalConnections = (*cell.numRequiredAdditionalConnections + MAX_CELL_BONDS + 1) % (MAX_CELL_BONDS + 1);
    }
    cell.energy = std::min(std::max(cell.energy, 50.0f), 1050.0f);

    switch (cell.getCellFunctionType()) {
    case CellFunction_Constructor: {
        auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
        if (constructor.mode < 0) {
            constructor.mode = 0;
        }
        if (constructor.maxConnections) {
            constructor.maxConnections = (*constructor.maxConnections + MAX_CELL_BONDS + 1) % (MAX_CELL_BONDS + 1);
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

void _GenomeEditorWindow::scheduleAddTab(GenomeDescription const& genome)
{
    TabData newTab;
    newTab.id = ++_tabSequenceNumber;
    newTab.genome = genome;
    _tabToAdd = newTab;
}

