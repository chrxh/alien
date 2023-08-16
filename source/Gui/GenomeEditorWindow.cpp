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
#include "Tooltips.h"

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
                AlienImGui::ComboParameters().name("Geometry").values(ShapeStrings).textWidth(ContentTextWidth).tooltip(Const::GenomeGeometryTooltip),
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
                .tooltip(Const::GenomeConnectionDistanceTooltip),
            tab.genome.info.connectionDistance);
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Stiffness").format("%.2f").step(0.05f).textWidth(ContentTextWidth).tooltip(Const::GenomeStiffnessTooltip),
            tab.genome.info.stiffness);
        if (tab.genome.info.shape == ConstructionShape_Custom) {
            table.next();
            AlienImGui::AngleAlignmentCombo(
                AlienImGui::AngleAlignmentComboParameters().name("Angle alignment").textWidth(ContentTextWidth).tooltip(Const::GenomeAngleAlignmentTooltip),
                tab.genome.info.angleAlignment);
        }
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Single construction").textWidth(ContentTextWidth).tooltip(Const::GenomeSingleConstructionTooltip),
            tab.genome.info.singleConstruction);
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Separate construction").textWidth(ContentTextWidth).tooltip(Const::GenomeSeparationConstructionTooltip),
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
        if (AlienImGui::CellFunctionCombo(
                AlienImGui::CellFunctionComboParameters().name("Function").textWidth(ContentTextWidth).tooltip(Const::getCellFunctionTooltip(type)), type)) {
            applyNewCellFunction(cell, type);
        }
        table.next();
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(ContentTextWidth).tooltip(Const::ColorTooltip), cell.color);
        if (!isFirstOrLast) {
            table.next();
            auto referenceAngle = shapeGeneratorResult ? shapeGeneratorResult->angle : cell.referenceAngle;
            if (AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Angle").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::AngleTooltip), referenceAngle)) {
                updateGeometry(tab.genome, tab.genome.info.shape);
                tab.genome.info.shape = ConstructionShape_Custom;
            }
            cell.referenceAngle = referenceAngle;
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::EnergyTooltip), cell.energy);
        table.next();
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters().name("Execution number").textWidth(ContentTextWidth).tooltip(Const::ExecutionNumberTooltip),
            cell.executionOrderNumber);
        table.next();
        AlienImGui::InputOptionalInt(
            AlienImGui::InputIntParameters().name("Input execution number").textWidth(ContentTextWidth).tooltip(Const::InputExecutionNumberTooltip),
            cell.inputExecutionOrderNumber);
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Block output").textWidth(ContentTextWidth).tooltip(Const::BlockOutputTooltip), cell.outputBlocked);
        table.next();
        auto numRequiredAdditionalConnections =
            shapeGeneratorResult ? shapeGeneratorResult->numRequiredAdditionalConnections : cell.numRequiredAdditionalConnections;
        if (AlienImGui::InputOptionalInt(
                AlienImGui::InputIntParameters().name("Required connections").textWidth(ContentTextWidth).tooltip(Const::RequiredConnectionsTooltip),
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
                    .tooltip(Const::TransmitterEnergyDistributionTooltip),
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
                        .tooltip(Const::ConstructorActivationModeTooltip),
                    constructorMode)) {
                constructor.mode = constructorMode;
            }
            if (constructorMode == 1) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Interval").textWidth(ContentTextWidth).tooltip(Const::ConstructorIntervalTooltip), constructor.mode);
                if (constructor.mode < 0) {
                    constructor.mode = 0;
                }
            }
            table.next();
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters()
                    .name("Offspring activation time")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::ConstructorOffspringActivationTime),
                constructor.constructionActivationTime);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #1")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::ConstructorAngle1Tooltip),
                constructor.constructionAngle1);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #2")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::ConstructorAngle2Tooltip),
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
                        .tooltip(Const::SensorModeTooltip),
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
                    AlienImGui::InputFloatParameters().name("Scan angle").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::SensorScanAngleTooltip),
                    *sensor.fixedAngle);
            }
            table.next();
            AlienImGui::ComboColor(
                AlienImGui::ComboColorParameters().name("Scan color").textWidth(ContentTextWidth).tooltip(Const::SensorScanColorTooltip), sensor.color);

            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Min density")
                    .format("%.2f")
                    .step(0.05f)
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::SensorMinDensityTooltip),
                sensor.minDensity);
        } break;
        case CellFunction_Nerve: {
            auto& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
            bool pulseGeneration = nerve.pulseMode > 0;
            table.next();
            if (AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters().name("Generate pulses").textWidth(ContentTextWidth).tooltip(Const::NerveGeneratePulsesTooltip),
                    pulseGeneration)) {
                nerve.pulseMode = pulseGeneration ? 1 : 0;
            }
            if (pulseGeneration) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Pulse interval").textWidth(ContentTextWidth).tooltip(Const::NervePulseIntervalTooltip),
                    nerve.pulseMode);
                bool alternation = nerve.alternationMode > 0;
                table.next();
                if (AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(ContentTextWidth).tooltip(Const::NerveAlternatingPulsesTooltip),
                        alternation)) {
                    nerve.alternationMode = alternation ? 1 : 0;
                }
                if (alternation) {
                    table.next();
                    AlienImGui::InputInt(
                        AlienImGui::InputIntParameters().name("Pulses per phase").textWidth(ContentTextWidth).tooltip(Const::NervePulsesPerPhase),
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
                    .tooltip(Const::AttackerEnergyDistributionTooltip),
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
                    .tooltip(Const::InjectorModeTooltip),
                injector.mode);
        } break;
        case CellFunction_Muscle: {
            auto& muscle = std::get<MuscleGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .values({"Movement", "Expansion and contraction", "Bending"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::MuscleModeTooltip),
                muscle.mode);
        } break;
        case CellFunction_Defender: {
            auto& defender = std::get<DefenderGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .values({"Anti-attacker", "Anti-injector"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::DefenderModeTooltip),
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
        AlienImGui::HelpMarker(Const::SubGenomeTooltip);
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

