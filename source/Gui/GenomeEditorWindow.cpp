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
#include "EngineInterface/ShapeGenerator.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "EditorModel.h"
#include "GenericFileDialogs.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "OverlayMessageController.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto const ContentTextWidth = 190.0f;
    auto const DynamicTableColumnWidth = 300.0f;
    auto const WeightsAndBiasTextWidth = 100.0f;
    auto const WeightsAndBiasSelectionTextWidth = 400.0f;
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
    if (_tabDatas.size() == 1 && _tabDatas.front().genome.cells.empty()) {
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
    std::string
    generateShortDescription(int index, CellGenomeDescription const& cell, std::optional<ShapeGeneratorResult> const& shapeGeneratorResult, bool isFirstOrLast)
    {
        auto result = "No. " + std::to_string(index + 1) + ", Type: " + Const::CellFunctionToStringMap.at(cell.getCellFunctionType())
            + ", Color: " + std::to_string(cell.color);
        if (!isFirstOrLast) {
            auto referenceAngle = shapeGeneratorResult ? shapeGeneratorResult->angle : cell.referenceAngle;
            result += ", Angle: " + StringHelper::format(referenceAngle, 1);
        }
        result += ", Energy: " + StringHelper::format(cell.energy, 1);
        return result;
    }
}

void _GenomeEditorWindow::processIntern()
{
    processToolbar();
    processEditor();
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
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedGenome = GenomeDescriptionConverter::convertDescriptionToBytes(selectedTab.genome);
    }
    AlienImGui::Tooltip("Copy genome");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
        onAddNode();
    }
    AlienImGui::Tooltip("Add cell");

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome.cells.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
        onDeleteNode();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete cell");

    ImGui::SameLine();
    auto& selectedNode = selectedTab.selectedNode;
    ImGui::BeginDisabled(!(selectedNode && *selectedNode > 0));
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_UP)) {
        onNodeDecreaseSequenceNumber();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Decrease sequence number of selected cell");

    ImGui::SameLine();
    ImGui::BeginDisabled(!(selectedNode && *selectedNode < selectedTab.genome.cells.size() - 1));
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_DOWN)) {
        onNodeIncreaseSequenceNumber();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Increase sequence number of selected cell");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS_SQUARE)) {
        _expandNodes = true;
    }
    AlienImGui::Tooltip("Expand all cells");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS_SQUARE)) {
        _expandNodes = false;
    }
    AlienImGui::Tooltip("Collapse all cells");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SEEDLING)) {
        onCreateSpore();
    }
    AlienImGui::Tooltip("Create a spore with current genome");

    AlienImGui::Separator();
}

void _GenomeEditorWindow::processEditor()
{
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

namespace
{
    class DynamicTableLayout
    {
    public:
        bool begin()
        {
            _columns = std::max(toInt(ImGui::GetContentRegionAvail().x / scale(DynamicTableColumnWidth)), 1);
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
                AlienImGui::VerticalSeparator();
                ImGui::SameLine();
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

void _GenomeEditorWindow::processTab(TabData& tab)
{
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(_previewHeight)), true)) {
        AlienImGui::Group("General properties");
        processGenomeHeader(tab);

        AlienImGui::Group("Construction sequence");
        processConstructionSequence(tab);
    }
    ImGui::EndChild();

    ImGui::Button("", ImVec2(-1, StyleRepository::getInstance().scale(5.0f)));
    if (ImGui::IsItemActive()) {
        _previewHeight -= ImGui::GetIO().MouseDelta.y;
    }

    AlienImGui::Group("Preview (reference configuration)");
    if (ImGui::BeginChild("##child4", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        showPreview(tab);
    }
    ImGui::EndChild();
}

void _GenomeEditorWindow::processGenomeHeader(TabData& tab)
{
    DynamicTableLayout table;
    if (table.begin()) {
        std::vector ShapeStrings = {"Custom"s, "Segment"s, "Triangle"s, "Rectangle"s, "Hexagon"s, "Loop"s, "Tube"s, "Lolli"s};
        auto origShape = tab.genome.info.shape;
        if (AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Geometry")
                    .values(ShapeStrings)
                    .textWidth(ContentTextWidth)
                    .tooltip("A genome describes a network of connected cells. On the one hand, there is the option to select a pre-defined geometry (e.g. "
                             "triangle or hexagon). Then, the cells encoded in the genome are generated along this geometry and connected together "
                             "appropriately. On the other hand, it is also possible to define custom geometries by setting an angle between predecessor and "
                             "successor cells for each cell (except for the first and last in the sequence)."),
                tab.genome.info.shape)) {
            updateGeometry(tab.genome, origShape);
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Connection distance")
                .format("%.2f")
                .step(0.05f)
                .textWidth(ContentTextWidth)
                .tooltip("The spatial distance between each cell and its predecessor cell in the genome sequence is determined here."),
            tab.genome.info.connectionDistance);
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Stiffness")
                .format("%.2f")
                .step(0.05f)
                .textWidth(ContentTextWidth)
                .tooltip("This value sets the stiffness for the entire encoded cell network."),
            tab.genome.info.stiffness);
        if (tab.genome.info.shape == ConstructionShape_Custom) {
            table.next();
            AlienImGui::AngleAlignmentCombo(
                AlienImGui::AngleAlignmentComboParameters()
                    .name("Angle alignment")
                    .textWidth(ContentTextWidth)
                    .tooltip("Triples of connected cells within a network have specific spatial angles relative to each other. These angles are guided by the "
                             "reference angles encoded in the cells. With this setting, it is optionally possible to specify that the reference angles must "
                             "only be multiples of certain values. This allows for greater stability of the created networks, as the angles would otherwise be "
                             "more susceptible to external influences. Choosing 60 degrees is recommended here, as it allows for the accurate representation "
                             "of most geometries."),
                tab.genome.info.angleAlignment);
        }
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Single construction")
                .textWidth(ContentTextWidth)
                .tooltip("This determines whether the encoded cell network in the genome should be constructed by the corresponding constructor cell only once "
                         "or multiple times."),
            tab.genome.info.singleConstruction);
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Separate construction")
                .textWidth(ContentTextWidth)
                .tooltip("Here, one can configure whether the encoded cell network in the genome should be detached from the constructor cell once it has been "
                         "fully constructed. Disabling this property is useful for encoding growing structures (such as plant-like species) or creature body "
                         "parts."),
            tab.genome.info.separateConstruction);
        table.end();
    }
    validationAndCorrection(tab.genome.info);
}

namespace 
{
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

void _GenomeEditorWindow::processConstructionSequence(TabData& tab)
{
    int index = 0;

    auto shapeGenerator = ShapeGeneratorFactory::create(tab.genome.info.shape);
    for (auto& cell : tab.genome.cells) {
        auto isFirstOrLast = index == 0 || index == tab.genome.cells.size() - 1;
        std::optional<ShapeGeneratorResult> shapeGeneratorResult =
            shapeGenerator ? std::make_optional(shapeGenerator->generateNextConstructionData()) : std::nullopt;

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

        if (_expandNodes) {
            ImGui::SetNextTreeNodeOpen(*_expandNodes);
        }
        auto treeNodeOpen =
            ImGui::TreeNodeEx((generateShortDescription(index, cell, shapeGeneratorResult, isFirstOrLast) + "###").c_str(), flags);
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
            processNode(tab, cell, shapeGeneratorResult, isFirstOrLast);
            if (origCell != cell) {
                tab.selectedNode = index;
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
        ++index;
    }
    _expandNodes.reset();
}

void _GenomeEditorWindow::processNode(
    TabData& tab,
    CellGenomeDescription& cell,
    std::optional<ShapeGeneratorResult> const& shapeGeneratorResult,
    bool isFirstOrLast)
{
    auto type = cell.getCellFunctionType();

    DynamicTableLayout table;
    if (table.begin()) {
        auto functionTooltip = [&] {
            switch (type) {
            case CellFunction_Neuron:
                return std::string("[Tooltip will be added.]");
            case CellFunction_Transmitter:
                return std::string(
                    "Transmitter cells are designed to transport energy. This is important, for example, to supply constructor cells with energy or to "
                    "support attacked cells. The energy transport works as follows: A part of the excess energy of the own cell and the directly connected "
                    "cells is collected and transferred to other cells in the vicinity. A cell has excess energy when it exceeds a defined normal value (see "
                    "simulation parameter 'Normal energy' in 'Cell life cycle'). Transmitter cells do not need an activation but they also transport the "
                    "activity states from input.");
            case CellFunction_Constructor:
                return std::string("[Tooltip will be added.]");
            case CellFunction_Sensor:
                return std::string("Sensor cells scan their environment for concentrations of cells of a certain color and provide distance and angle to the "
                                   "closest match.");
            case CellFunction_Nerve:
                return std::string("[Tooltip will be added.]");
            case CellFunction_Attacker:
                return std::string("[Tooltip will be added.]");
            case CellFunction_Injector:
                return std::string("[Tooltip will be added.]");
            case CellFunction_Muscle:
                return std::string("[Tooltip will be added.]");
            case CellFunction_Defender:
                return std::string("[Tooltip will be added.]");
            default:
                return std::string(
                    "Cells can possess a specific function that enables them to, for example, perceive their environment, process information, or "
                    "take action. If you choose a cell function this tooltip will be updated to provide the corresponding information.");
            }
        }();

        if (AlienImGui::CellFunctionCombo(
                AlienImGui::CellFunctionComboParameters()
                    .name("Function")
                    .textWidth(ContentTextWidth).tooltip(functionTooltip),
                type)) {
            applyNewCellFunction(cell, type);
        }
        table.next();
        AlienImGui::ComboColor(
            AlienImGui::ComboColorParameters()
                .name("Color")
                .textWidth(ContentTextWidth)
                .tooltip("On the one hand, the cell color can be used to define own types of cells that are subject to different rules. For this purpose, the "
                         "simulation parameters can be specified depending on the color. For example, one could define that green cells are particularly good "
                         "at absorbing energy particles, while other cell colors are better at attacking foreign cells.\nOn the other hand, cell color also "
                         "plays a role in perception. Sensor cells are dedicated to a specific color and can only detect the corresponding cells."),
            cell.color);
        if (!isFirstOrLast) {
            table.next();
            auto referenceAngle = shapeGeneratorResult ? shapeGeneratorResult->angle : cell.referenceAngle;
            if (AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters()
                        .name("Angle")
                        .textWidth(ContentTextWidth)
                        .format("%.1f")
                        .tooltip("The angle between the predecessor and successor cell can be specified here. Please note that the shown angle here is shifted "
                                 "by 180 degrees for convenience. In other words, a value of 0 actually corresponds to an angle of 180 degrees, i.e. a straight segment."),
                    referenceAngle)) {
                updateGeometry(tab.genome, tab.genome.info.shape);
                tab.genome.info.shape = ConstructionShape_Custom;
            }
            cell.referenceAngle = referenceAngle;
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Energy")
                .textWidth(ContentTextWidth)
                .format("%.1f")
                .tooltip("The energy that the cell should receive after its creation. The larger this value is, the more energy the constructor cell must expend to create it."),
            cell.energy);
        table.next();
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Execution number")
                .textWidth(ContentTextWidth)
                .tooltip("The functions of cells can be executed in a specific sequence determined by this number. The values are limited between 0 and 5 and "
                         "follow a modulo 6 logic. For example, a cell with an execution number of 0 will be executed at time points 0, 6, 12, 18, etc. A cell "
                         "with an execution number of 1 will be shifted by one, i.e. executed at 1, 7, 13, 19, etc. This time offset enables the orchestration "
                         "of cell functions. A muscle cell, for instance, requiring input from a neuron cell, should then be executed one time step later."),
            cell.executionOrderNumber);
        table.next();
        AlienImGui::InputOptionalInt(
            AlienImGui::InputIntParameters()
                .name("Input execution number")
                .textWidth(ContentTextWidth)
                .tooltip(
                    "A functioning organism requires cells to collaborate. This can involve sensor cells that perceive the environment, neuron cells that "
                    "process information, muscle cells that perform movements, and so on. These various cell functions often require input and produce an "
                    "output. Both input and output are based on the cell's activity states. The process for updating is performed in two steps:\n\n1) When a "
                    "cell function is executed, the activity states are first updated. This involves reading the activity states of all connected cells "
                    "whose 'execution number' matches the specified 'input execution number', summing them up, and then setting the result to the "
                    "activity states for the considered cell.\n\n2) The cell function is executed and can use the cell's activity states as input. "
                    "The output is used to update the activity states again.\n\nSetting an 'input execution number' is optional. If none is set, the cell can "
                    "receive no input."),
            cell.inputExecutionOrderNumber);
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Block output")
                .textWidth(ContentTextWidth)
                .tooltip("Activating this toggle, the cell's output can be locked, preventing any other cell from utilizing it as input."),
            cell.outputBlocked);
        table.next();
        auto numRequiredAdditionalConnections =
            shapeGeneratorResult ? shapeGeneratorResult->numRequiredAdditionalConnections : cell.numRequiredAdditionalConnections;
        if (AlienImGui::InputOptionalInt(
                AlienImGui::InputIntParameters()
                    .name("Required connections")
                    .textWidth(ContentTextWidth)
                    .tooltip(
                        "By default, cells in the genome sequence are automatically connected to all neighboring cells belonging to the same genome when they "
                        "are created. However, this can pose a challenge because the constructed cells need time to fold into their desired positions. If the "
                        "current spatial location of the constructor cell is unfavorable, the newly formed cell might not be connected to the desired cells, "
                        "for instance, due to being too far away. An better approach would involve delaying the construction process until a desired number of "
                        "neighboring cells from the same genome are in the direct vicinity. This number of cells can be optionally set here.\nIt is important "
                        "to note that the predecessor cell is not counted for the 'required connections.' For example, a value of 2 means that the cell to be "
                        "constructed will only be created when there are at least 2 already constructed cells (excluding the predecessor cell) available for "
                        "potential connections. If the condition is not met, the construction process is postponed."),
                numRequiredAdditionalConnections)) {
            updateGeometry(tab.genome, tab.genome.info.shape);
            tab.genome.info.shape = ConstructionShape_Custom;
        }
        cell.numRequiredAdditionalConnections = numRequiredAdditionalConnections;

        switch (type) {
        case CellFunction_Neuron: {
        } break;
        case CellFunction_Transmitter: {
            auto& transmitter = std::get<TransmitterGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Energy distribution")
                    .values({"Connected cells", "Transmitters and constructors"})
                    .textWidth(ContentTextWidth)
                    .tooltip("There are two ways to control the energy distribution, which is set "
                             "here:\n\n" ICON_FA_CHEVRON_RIGHT " Connected cells: "
                             "In this case the energy will be distributed evenly across all connected and connected-connected cells.\n\n" ICON_FA_CHEVRON_RIGHT
                             " Transmitters and constructors: "
                             "Here, the energy will be transferred to spatially nearby constructors or other transmitter cells within the same cell "
                             "network. If multiple such transmitter cells are present at certain distances, energy can be transmitted over greater distances, "
                             "for example, from attacker cells to constructor cells."),
                transmitter.mode);
        } break;
        case CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);

            int constructorMode = constructor.mode == 0 ? 0 : 1;
            table.next();
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Activation mode")
                        .textWidth(ContentTextWidth)
                        .values({"Manual", "Automatic"})
                        .tooltip("There are 2 modes available for controlling constructor cells:\n\n" ICON_FA_CHEVRON_RIGHT " Manual: The construction process is only triggered when "
                                 "there is activity in channel #0.\n\n" ICON_FA_CHEVRON_RIGHT " Automatic: The construction process is automatically triggered at regular intervals. "
                                 "Activity in channel #0 is not necessary.\n\n In both cases, if there is not enough energy available for the cell being "
                                 "created, the construction process will pause until the next triggering."),
                    constructorMode)) {
                constructor.mode = constructorMode;
            }
            if (constructorMode == 1) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters()
                        .name("Interval")
                        .textWidth(ContentTextWidth)
                        .tooltip("This value specifies the time interval for automatic triggering of the constructor cell. It is given in multiples "
                                 "of 6 (which is a complete execution cycle). This means that a value of 1 indicates that the constructor cell will be activated "
                                 "every 6 time steps"),
                    constructor.mode);
                if (constructor.mode < 0) {
                    constructor.mode = 0;
                }
            }
            table.next();
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters()
                    .name("Offspring activation time")
                    .textWidth(ContentTextWidth)
                    .tooltip("When a new cell network has been fully constructed by a constructor cell, you can set the time until activation here. This is "
                             "especially useful when the offspring should not become active immediately, for example, to prevent it from attacking."),
                constructor.constructionActivationTime);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #1")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip("By default, when the constructor cell initiates a new construction, the new cell is created in the area with the most available "
                             "space. This angle specifies the deviation from that rule."),
                constructor.constructionAngle1);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #2")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip("This value determines the angle from the last constructed cell to the second-last constructed cell and the constructor cell. The "
                             "effects can be best observed in the preview (in the lower part of the editor)."),
                constructor.constructionAngle2);
        } break;
        case CellFunction_Sensor: {
            auto& sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
            auto sensorMode = sensor.getSensorMode();

            table.next();
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Mode")
                        .textWidth(ContentTextWidth)
                        .values({"Scan vicinity", "Scan specific direction"})
                        .tooltip("Sensors can operate in 2 modes:\n\n" ICON_FA_CHEVRON_RIGHT
                                 " Scan vicinity: In this mode, the entire nearby area is scanned (typically "
                                 "within a radius of several 100 units). The scan radius can be adjusted via a simulation parameter (see 'Range' in the sensor "
                                 "settings).\n\n" ICON_FA_CHEVRON_RIGHT
                                 " Scan specific direction: In this mode, the scanning process is restricted to a particular direction. The "
                                 "direction is specified as an angle."),
                    sensorMode)) {
                if (sensorMode == SensorMode_Neighborhood) {
                    sensor.fixedAngle.reset();
                } else {
                    sensor.fixedAngle = 0.0f;
                }
            }
            if (sensorMode == SensorMode_FixedAngle) {
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters()
                        .name("Scan angle")
                        .textWidth(ContentTextWidth)
                        .format("%.1f")
                        .tooltip("The angle can be determined here in which direction the scanning process should take place. An angle of 0 means that the "
                                 "scan should occur in the direction derived from the predecessor cell (the cell from which the activity input originates) "
                                 "towards the sensor cell."),
                    *sensor.fixedAngle);
            }
            table.next();
            AlienImGui::ComboColor(
                AlienImGui::ComboColorParameters().name("Scan color").textWidth(ContentTextWidth).tooltip("Specifies the color of the cells to search for."),
                sensor.color);

            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Min density")
                    .format("%.2f")
                    .step(0.05f)
                    .textWidth(ContentTextWidth)
                    .tooltip("The minimum density to search for a cell concentration of a specific color. This value ranges between 0 and 1. It controls the "
                             "sensitivity of the sensor. Typically, very few cells of the corresponding color are already detected with a value of 0.1."),
                sensor.minDensity);
        } break;
        case CellFunction_Nerve: {
            auto& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
            bool pulseGeneration = nerve.pulseMode > 0;
            table.next();
            if (AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Generate pulses")
                        .textWidth(ContentTextWidth)
                        .tooltip("By default, a nerve cell forwards activity states by receiving activity as input from connected cells (and summing it if "
                                 "there are multiple cells) and directly providing it as output to other cells. Independently of this, you can specify here "
                                 "that it also generates an activity pulse in channel #0 at regular intervals. This can be used to trigger other sensor cells, "
                                 "attacker cells, etc."),
                    pulseGeneration)) {
                nerve.pulseMode = pulseGeneration ? 1 : 0;
            }
            if (pulseGeneration) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters()
                        .name("Pulse interval")
                        .textWidth(ContentTextWidth)
                        .tooltip("The intervals between two pulses can be set here. It is specified in cycles, which corresponds to 6 time steps each."),
                    nerve.pulseMode);
                bool alternation = nerve.alternationMode > 0;
                table.next();
                if (AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters()
                            .name("Alternating pulses")
                            .textWidth(ContentTextWidth)
                            .tooltip("By default, the generated pulses consist of a positive value in channel #0. When 'Alternating pulses' is enabled, the "
                                     "sign of this value alternates at specific time intervals. This can be used, for example, to easily create control "
                                     "signals for back-and-forth movements or bending in muscle cells."),
                        alternation)) {
                    nerve.alternationMode = alternation ? 1 : 0;
                }
                if (alternation) {
                    table.next();
                    AlienImGui::InputInt(
                        AlienImGui::InputIntParameters()
                            .name("Pulses per phase")
                            .textWidth(ContentTextWidth)
                            .tooltip("This value indicates the number of pulses until a sign changes in the generated activity."),
                        nerve.alternationMode);
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
                    .textWidth(ContentTextWidth)
                    .tooltip("Attacker cells can distribute the acquired energy through two different methods. The energy distribution is analogous to "
                             "transmitter cells. \n\n" ICON_FA_CHEVRON_RIGHT " Connected cells: In this case the energy will be distributed evenly across all "
                             "connected and connected-connected cells.\n\n" ICON_FA_CHEVRON_RIGHT
                             " Transmitters and constructors: Here, the energy will be transferred to spatially nearby constructors or other transmitter cells "
                             "within the same cell network. If multiple such transmitter cells are present at certain distances, energy can be transmitted "
                             "over greater distances, for example, from attacker cells to constructor cells."),
                attacker.mode);
        } break;
        case CellFunction_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .textWidth(ContentTextWidth)
                    .values({"Cells under construction", "All Cells"})
                    .tooltip(
                        "Injector cells can overwrite their genome into other constructor or injector cells. To do this, they need to be activated via channel "
                        "#0, remain in close proximity to the target cell for a certain minimum duration, and, in the case of another constructor cell, "
                        "its construction process must not have started yet. There are two modes to choose from:\n\n" ICON_FA_CHEVRON_RIGHT
                        " Cells under construction: Only cells which are under construction can be infected. This mode is useful when an organism wants to "
                        "inject its genome into another own constructor cell (e.g. to build a spore).\n\n" ICON_FA_CHEVRON_RIGHT
                        " All Cells: In this mode there are no restrictions, e.g. any other constructor or injector cell can be infected."),
                injector.mode);
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
    if (ImGui::BeginChild("##", ImVec2(width, scale(60.0f)), true)) {
        AlienImGui::MonospaceText(content);
        AlienImGui::HelpMarker("If a constructor or injector cell is encoded in a genome, that cell can itself contain another genome. This sub-genome can "
                               "describe additional body parts or branching of the creature, for instance. Furthermore, sub-genomes can in turn possess further "
                               "sub-sub-genomes, etc. To insert a sub-genome here by clicking on 'Paste', one must have previously copied one to the clipboard. "
                               "This can be done using the 'Copy genome' button in the toolbar. This action copies the entire genome from the current tab to "
                               "the clipboard. If you want to create self-replication, you must not insert a sub-genome; instead, you switch it to the "
                               "'self-copy' mode. In this case, the constructor's sub-genome refers to its superordinate genome.");
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
    auto& cells = tabData.genome.cells;

    CellGenomeDescription newNode;
    if (tabData.selectedNode) {
        newNode.color = cells.at(*tabData.selectedNode).color;
        cells.insert(cells.begin() + *tabData.selectedNode + 1, newNode);
        ++(*tabData.selectedNode);
    } else {
        if (!cells.empty()) {
            newNode.color = cells.back().color;
        }
        cells.emplace_back(newNode);
        tabData.selectedNode = toInt(cells.size() - 1);
    }
}

void _GenomeEditorWindow::onDeleteNode()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);
    auto& cells = tabData.genome.cells;

    if (tabData.selectedNode) {
        cells.erase(cells.begin() + *tabData.selectedNode);
        if (*tabData.selectedNode == toInt(cells.size())) {
            if (--(*tabData.selectedNode) < 0) {
                tabData.selectedNode.reset();
            }
        }
    } else {
        cells.pop_back();
        if (!cells.empty()) {
            tabData.selectedNode = toInt(cells.size() - 1);
        }
    }
    _expandNodes = false;
}

void _GenomeEditorWindow::onNodeDecreaseSequenceNumber()
{
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selectedNode = selectedTab.selectedNode;
    std::swap(selectedTab.genome.cells.at(*selectedNode), selectedTab.genome.cells.at(*selectedNode - 1));
    --(*selectedNode);
    _expandNodes = false;
}

void _GenomeEditorWindow::onNodeIncreaseSequenceNumber()
{
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selectedNode = selectedTab.selectedNode;
    std::swap(selectedTab.genome.cells.at(*selectedNode), selectedTab.genome.cells.at(*selectedNode + 1));
    ++(*selectedNode);
    _expandNodes = false;
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
                    .setEnergy(parameter.cellNormalEnergy[_editorModel->getDefaultColorCode()] * (genomeDesc.cells.size() * 2 + 1))
                    .setStiffness(1.0f)
                    .setMaxConnections(6)
                    .setExecutionOrderNumber(0)
                    .setColor(_editorModel->getDefaultColorCode())
                    .setCellFunction(ConstructorDescription().setGenome(genome));
    auto data = DataDescription().addCell(cell);
    _simController->addAndSelectSimulationData(data);
    _editorModel->update();

    printOverlayMessage("Spore created");
}

void _GenomeEditorWindow::showPreview(TabData& tab)
{
    auto const& genome = _tabDatas.at(_selectedTabIndex).genome;
    auto preview = PreviewDescriptionConverter::convert(genome, tab.selectedNode, _simController->getSimulationParameters());
    if (AlienImGui::ShowPreviewDescription(preview, _previewZoom, tab.selectedNode)) {
        _nodeIndexToJump = tab.selectedNode;
    }
}

void _GenomeEditorWindow::validationAndCorrection(GenomeHeaderDescription& info) const
{
    info.stiffness = std::max(0.0f, std::min(1.0f, info.stiffness));
    info.connectionDistance = std::max(0.5f, std::min(1.5f, info.connectionDistance));
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

void _GenomeEditorWindow::updateGeometry(GenomeDescription& genome, ConstructionShape shape)
{
    auto shapeGenerator = ShapeGeneratorFactory::create(shape);
    if (!shapeGenerator) {
        return;
    }
    genome.info.angleAlignment = shapeGenerator->getConstructorAngleAlignment();
    for (auto& cell : genome.cells) {
        auto shapeGenerationResult = shapeGenerator->generateNextConstructionData();
        cell.referenceAngle = shapeGenerationResult.angle;
        cell.numRequiredAdditionalConnections = shapeGenerationResult.numRequiredAdditionalConnections;
    }
}

