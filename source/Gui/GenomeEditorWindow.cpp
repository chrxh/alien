#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <ImFileDialog.h>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/PreviewDescriptionService.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/ShapeGenerator.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "DelayedExecutionController.h"
#include "EditorModel.h"
#include "GenericFileDialogs.h"
#include "MessageDialog.h"
#include "OverlayMessageController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "HelpStrings.h"
#include "UploadSimulationDialog.h"
#include "ChangeColorDialog.h"

namespace
{
    auto const PreviewHeight = 300.0f;
    auto const ContentHeaderTextWidth = 215.0f;
    auto const ContentTextWidth = 190.0f;
    auto const DynamicTableHeaderColumnWidth = 335.0f;
    auto const DynamicTableColumnWidth = 308.0f;
    auto const SubWindowRightMargin = 0.0f;
}

_GenomeEditorWindow ::_GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController)
    : _AlienWindow("Genome editor", "windows.genome editor", false)
    , _editorModel(editorModel)
    , _simController(simulationController)
{
    _tabDatas = {TabData()};

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getString("windows.genome editor.starting path", path.string());
    _previewHeight = GlobalSettings::getInstance().getFloat("windows.genome editor.preview height", scale(PreviewHeight));
    _changeColorDialog =
        std::make_shared<_ChangeColorDialog>([&] { return getCurrentGenome(); }, [&](GenomeDescription const& genome) { setCurrentGenome(genome); });
}

_GenomeEditorWindow::~_GenomeEditorWindow()
{
    GlobalSettings::getInstance().setString("windows.genome editor.starting path", _startingPath);
    GlobalSettings::getInstance().setFloat("windows.genome editor.preview height", _previewHeight);
}

void _GenomeEditorWindow::registerCyclicReferences(UploadSimulationDialogWeakPtr const& uploadSimulationDialog)
{
    _uploadSimulationDialog = uploadSimulationDialog;
}

void _GenomeEditorWindow::openTab(GenomeDescription const& genome, bool openGenomeEditorIfClosed)
{
    if (openGenomeEditorIfClosed) {
        setOn(false);
        delayedExecution([this] { setOn(true); });
    }
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
    _changeColorDialog->process();
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
    if (AlienImGui::ToolbarButton(ICON_FA_UPLOAD)) {
        onUploadGenome();
    }
    AlienImGui::Tooltip("Share your genome with other users:\nYour current genome will be uploaded to the server and made visible in the browser.");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedGenome = GenomeDescriptionService::convertDescriptionToBytes(selectedTab.genome);
        printOverlayMessage("Genome copied");
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
    ImGui::BeginDisabled(selectedTab.genome.cells.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_EXPAND_ARROWS_ALT)) {
        _expandNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Expand all cells");

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome.cells.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_COMPRESS_ARROWS_ALT)) {
        _expandNodes = false;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Collapse all cells");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome.cells.empty());
    if (AlienImGui::ToolbarButton(ICON_FA_PALETTE)) {
        _changeColorDialog->open();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Change the color of all cells with a specific color");

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

void _GenomeEditorWindow::processTab(TabData& tab)
{
    _previewHeight = std::min(ImGui::GetContentRegionAvail().y - 10.0f, std::max(10.0f, _previewHeight));
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - _previewHeight), true)) {
        AlienImGui::Group("General properties");
        processGenomeHeader(tab);

        AlienImGui::Group("Construction sequence");
        processConstructionSequence(tab);
    }
    ImGui::EndChild();

    AlienImGui::MovableSeparator(_previewHeight);

    AlienImGui::Group("Preview (reference configuration)");
    if (ImGui::BeginChild("##child4", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        showPreview(tab);
    }
    ImGui::EndChild();
}

void _GenomeEditorWindow::processGenomeHeader(TabData& tab)
{
    AlienImGui::DynamicTableLayout table(DynamicTableHeaderColumnWidth);
    if (table.begin()) {
        std::vector ShapeStrings = {"Custom"s, "Segment"s, "Triangle"s, "Rectangle"s, "Hexagon"s, "Loop"s, "Tube"s, "Lolli"s, "Small lolli"s, "Zigzag"s};
        auto origShape = tab.genome.header.shape;
        if (AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Geometry").values(ShapeStrings).textWidth(ContentHeaderTextWidth).tooltip(Const::GenomeGeometryTooltip),
                tab.genome.header.shape)) {
            updateGeometry(tab.genome, origShape);
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Connection distance")
                .format("%.2f")
                .step(0.05f)
                .textWidth(ContentHeaderTextWidth)
                .tooltip(Const::GenomeConnectionDistanceTooltip),
            tab.genome.header.connectionDistance);
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Stiffness")
                .format("%.2f")
                .step(0.05f)
                .textWidth(ContentHeaderTextWidth)
                .tooltip(Const::GenomeStiffnessTooltip),
            tab.genome.header.stiffness);
        if (tab.genome.header.shape == ConstructionShape_Custom) {
            table.next();
            AlienImGui::AngleAlignmentCombo(
                AlienImGui::AngleAlignmentComboParameters()
                    .name("Angle alignment")
                    .textWidth(ContentHeaderTextWidth)
                    .tooltip(Const::GenomeAngleAlignmentTooltip),
                tab.genome.header.angleAlignment);
        }
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Separation").textWidth(ContentHeaderTextWidth).tooltip(Const::GenomeSeparationConstructionTooltip),
            tab.genome.header.separateConstruction);
        table.next();
        if (!tab.genome.header.separateConstruction) {
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Number of branches").textWidth(ContentHeaderTextWidth).tooltip(Const::GenomeNumBranchesTooltip),
                tab.genome.header.numBranches);
            table.next();
        }
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Repetitions per branch")
                .infinity(true)
                .textWidth(ContentHeaderTextWidth)
                .tooltip(Const::GenomeRepetitionsPerBranchTooltip),
            tab.genome.header.numRepetitions);
        if (tab.genome.header.numRepetitions > 1) {
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Concatenation angle #1")
                    .format("%.1f")
                    .textWidth(ContentHeaderTextWidth)
                    .tooltip(Const::GenomeConcatenationAngle1)
                ,
                tab.genome.header.concatenationAngle1);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Concatenation angle #2")
                    .format("%.1f")
                    .textWidth(ContentHeaderTextWidth)
                    .tooltip(Const::GenomeConcatenationAngle2),
                tab.genome.header.concatenationAngle2);
        }
        table.end();
    }
    validationAndCorrection(tab.genome.header);
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
            cell.cellFunction = ConstructorGenomeDescription().setGenome(GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()));
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
            cell.cellFunction = InjectorGenomeDescription().setGenome(GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()));
        } break;
        case CellFunction_Muscle: {
            cell.cellFunction = MuscleGenomeDescription();
        } break;
        case CellFunction_Defender: {
            cell.cellFunction = DefenderGenomeDescription();
        } break;
        case CellFunction_Reconnector: {
            cell.cellFunction = ReconnectorGenomeDescription();
        } break;
        case CellFunction_Detonator: {
            cell.cellFunction = DetonatorGenomeDescription();
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

    auto shapeGenerator = ShapeGeneratorFactory::create(tab.genome.header.shape);
    for (auto& cell : tab.genome.cells) {
        auto isFirst = index == 0;
        auto isLast = index == tab.genome.cells.size() - 1;
        auto isFirstOrLast = isFirst || isLast;
        std::optional<ShapeGeneratorResult> shapeGeneratorResult =
            shapeGenerator ? std::make_optional(shapeGenerator->generateNextConstructionData()) : std::nullopt;

        ImGui::PushID(index);

        float h, s, v;
        AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(h, s * 0.5f, v));
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_Framed;

        auto isNodeSelected = tab.selectedNode && *tab.selectedNode == index;
        if (isNodeSelected) {
            flags |= ImGuiTreeNodeFlags_Selected;
        } else {
            ImGui::PushStyleColor(ImGuiCol_Header, static_cast<ImVec4>(ImColor::HSV(0, 0, 0, 0)));
        }
        if (_nodeIndexToJump && *_nodeIndexToJump == index) {
            ImGui::SetScrollHereY();
            _nodeIndexToJump = std::nullopt;
        }

        if (_expandNodes) {
            ImGui::SetNextTreeNodeOpen(*_expandNodes);
        }
        ImGui::PushFont(StyleRepository::getInstance().getSmallBoldFont());
        auto treeNodeOpen =
            ImGui::TreeNodeEx((generateShortDescription(index, cell, shapeGeneratorResult, isFirstOrLast) + "###").c_str(), flags);
        ImGui::PopFont();
        if (!isNodeSelected) {
            ImGui::PopStyleColor();
        }
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
            processNode(tab, cell, shapeGeneratorResult, isFirst, isLast);
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
    bool isFirst,
    bool isLast)
{
    auto type = cell.getCellFunctionType();
    auto isFirstOrLast = isFirst || isLast;

    AlienImGui::DynamicTableLayout table(DynamicTableColumnWidth);
    if (table.begin()) {
        if (AlienImGui::CellFunctionCombo(
                AlienImGui::CellFunctionComboParameters().name("Function").textWidth(ContentTextWidth).tooltip(Const::getCellFunctionTooltip(type)), type)) {
            applyNewCellFunction(cell, type);
        }
        table.next();
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(ContentTextWidth).tooltip(Const::GenomeColorTooltip), cell.color);
        if (!isFirstOrLast) {
            table.next();
            auto referenceAngle = shapeGeneratorResult ? shapeGeneratorResult->angle : cell.referenceAngle;
            if (AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Angle").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::GenomeAngleTooltip), referenceAngle)) {
                updateGeometry(tab.genome, tab.genome.header.shape);
                tab.genome.header.shape = ConstructionShape_Custom;
            }
            cell.referenceAngle = referenceAngle;
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::GenomeEnergyTooltip), cell.energy);
        table.next();
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters().name("Execution number").textWidth(ContentTextWidth).tooltip(Const::GenomeExecutionNumberTooltip),
            cell.executionOrderNumber);
        table.next();
        AlienImGui::InputOptionalInt(
            AlienImGui::InputIntParameters().name("Input execution number").textWidth(ContentTextWidth).tooltip(Const::GenomeInputExecutionNumberTooltip),
            cell.inputExecutionOrderNumber);
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Block output").textWidth(ContentTextWidth).tooltip(Const::GenomeBlockOutputTooltip), cell.outputBlocked);
        table.next();
        auto numRequiredAdditionalConnections =
            shapeGeneratorResult ? shapeGeneratorResult->numRequiredAdditionalConnections : cell.numRequiredAdditionalConnections;
        if (!isFirst && numRequiredAdditionalConnections) {
            numRequiredAdditionalConnections = std::min(*numRequiredAdditionalConnections + 1, MAX_CELL_BONDS);
        }
        if (AlienImGui::InputOptionalInt(
                AlienImGui::InputIntParameters().name("Required connections").textWidth(ContentTextWidth).tooltip(Const::GenomeRequiredConnectionsTooltip),
                numRequiredAdditionalConnections)) {
            updateGeometry(tab.genome, tab.genome.header.shape);
            tab.genome.header.shape = ConstructionShape_Custom;
        }
        if (!isFirst && numRequiredAdditionalConnections) {
            numRequiredAdditionalConnections = *numRequiredAdditionalConnections - 1;
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
                    .tooltip(Const::GenomeTransmitterEnergyDistributionTooltip),
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
                        .tooltip(Const::GenomeConstructorActivationModeTooltip),
                    constructorMode)) {
                constructor.mode = constructorMode;
            }
            if (constructorMode == 1) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Interval").textWidth(ContentTextWidth).tooltip(Const::GenomeConstructorIntervalTooltip), constructor.mode);
                if (constructor.mode < 0) {
                    constructor.mode = 0;
                }
            }
            table.next();
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters()
                    .name("Offspring activation time")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeConstructorOffspringActivationTime),
                constructor.constructionActivationTime);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #1")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeConstructorAngle1Tooltip),
                constructor.constructionAngle1);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #2")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeConstructorAngle2Tooltip),
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
                        .tooltip(Const::GenomeSensorModeTooltip),
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
                    AlienImGui::InputFloatParameters().name("Scan angle").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::GenomeSensorScanAngleTooltip),
                    *sensor.fixedAngle);
            }
            table.next();
            AlienImGui::ComboOptionalColor(
                AlienImGui::ComboColorParameters().name("Scan color").textWidth(ContentTextWidth).tooltip(Const::GenomeSensorScanColorTooltip), sensor.restrictToColor);
            table.next();
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Restrict to other mutants")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeSensorRestrictToOtherMutantsTooltip),
                sensor.restrictToOtherMutants);

            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Min density")
                    .format("%.2f")
                    .step(0.05f)
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeSensorMinDensityTooltip),
                sensor.minDensity);
        } break;
        case CellFunction_Nerve: {
            auto& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
            bool pulseGeneration = nerve.pulseMode > 0;
            table.next();
            if (AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters().name("Generate pulses").textWidth(ContentTextWidth).tooltip(Const::GenomeNerveGeneratePulsesTooltip),
                    pulseGeneration)) {
                nerve.pulseMode = pulseGeneration ? 1 : 0;
            }
            if (pulseGeneration) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Pulse interval").textWidth(ContentTextWidth).tooltip(Const::GenomeNervePulseIntervalTooltip),
                    nerve.pulseMode);
                bool alternation = nerve.alternationMode > 0;
                table.next();
                if (AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(ContentTextWidth).tooltip(Const::GenomeNerveAlternatingPulsesTooltip),
                        alternation)) {
                    nerve.alternationMode = alternation ? 1 : 0;
                }
                if (alternation) {
                    table.next();
                    AlienImGui::InputInt(
                        AlienImGui::InputIntParameters().name("Pulses per phase").textWidth(ContentTextWidth).tooltip(Const::GenomeNervePulsesPerPhaseTooltip),
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
                    .tooltip(Const::GenomeAttackerEnergyDistributionTooltip),
                attacker.mode);
        } break;
        case CellFunction_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .textWidth(ContentTextWidth)
                    .values({"Only empty cells", "All cells"})
                    .tooltip(Const::GenomeInjectorModeTooltip),
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
                    .tooltip(Const::GenomeMuscleModeTooltip),
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
                    .tooltip(Const::GenomeDefenderModeTooltip),
                defender.mode);
        } break;
        case CellFunction_Reconnector: {
            auto& reconnector = std::get<ReconnectorGenomeDescription>(*cell.cellFunction);
            table.next();
            AlienImGui::ComboColor(
                AlienImGui::ComboColorParameters().name("Target color").textWidth(ContentTextWidth).tooltip(Const::GenomeReconnectorTargetColorTooltip),
                reconnector.color);
        } break;
        case CellFunction_Detonator: {
            table.next();
            auto& detonator = std::get<DetonatorGenomeDescription>(*cell.cellFunction);
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Countdown").textWidth(ContentTextWidth).tooltip(Const::GenomeDetonatorCountdownTooltip),
                detonator.countdown);
            detonator.countdown = std::min(65535, std::max(0, detonator.countdown));
        } break;
        }

        table.end();

        switch (type) {
        case CellFunction_Neuron: {
            auto& neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
            if (ImGui::TreeNodeEx("Neural network", ImGuiTreeNodeFlags_None)) {
                AlienImGui::NeuronSelection(
                    AlienImGui::NeuronSelectionParameters().rightMargin(SubWindowRightMargin), neuron.weights, neuron.biases, neuron.activationFunctions);
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
    auto width = ImGui::GetContentRegionAvail().x - scale(SubWindowRightMargin);
    if (ImGui::BeginChild("##", ImVec2(width, scale(60.0f)), true)) {
        AlienImGui::MonospaceText(content);
        AlienImGui::HelpMarker(Const::SubGenomeTooltip);
        if (AlienImGui::Button("Clear")) {
            desc.setGenome(GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()));
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Copy")) {
            _copiedGenome = desc.isMakeGenomeCopy() ? GenomeDescriptionService::convertDescriptionToBytes(tab.genome) : desc.getGenomeData();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!_copiedGenome.has_value());
        if (AlienImGui::Button("Paste")) {
            desc.genome = *_copiedGenome;
            printOverlayMessage("Genome pasted");
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (AlienImGui::Button("Edit")) {
            auto genomeToOpen = desc.isMakeGenomeCopy()
                ? tab.genome
                : GenomeDescriptionService::convertBytesToDescription(desc.getGenomeData());
            openTab(genomeToOpen, false);
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Set self-copy")) {
            desc.setMakeSelfCopy();
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
        if (!SerializerService::deserializeGenomeFromFile(genomeData, firstFilename.string())) {
            MessageDialog::getInstance().information("Open genome", "The selected file could not be opened.");
        } else {
            openTab(GenomeDescriptionService::convertBytesToDescription(genomeData), false);
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
            auto genomeData = GenomeDescriptionService::convertDescriptionToBytes(selectedTab.genome);
            if (!SerializerService::serializeGenomeToFile(firstFilename.string(), genomeData)) {
                MessageDialog::getInstance().information("Save genome", "The selected file could not be saved.");
            }
        });
}

void _GenomeEditorWindow::onUploadGenome()
{
    _uploadSimulationDialog.lock()->open(NetworkResourceType_Genome);
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
    auto pos = Viewport::getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto genomeDesc = getCurrentGenome();
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);

    auto parameter = _simController->getSimulationParameters();
    auto numNodes = GenomeDescriptionService::getNumNodesRecursively(genome, true);
    auto energy = parameter.cellNormalEnergy[_editorModel->getDefaultColorCode()] * toFloat(numNodes * 2 + 1);
    auto cell = CellDescription()
                    .setPos(pos)
                    .setEnergy(energy)
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
    auto preview = PreviewDescriptionService::convert(genome, tab.selectedNode, _simController->getSimulationParameters());
    if (AlienImGui::ShowPreviewDescription(preview, tab.previewZoom, tab.selectedNode)) {
        _nodeIndexToJump = tab.selectedNode;
    }
}

void _GenomeEditorWindow::validationAndCorrection(GenomeHeaderDescription& header) const
{
    header.stiffness = std::max(0.0f, std::min(1.0f, header.stiffness));
    header.connectionDistance = std::max(0.5f, std::min(1.5f, header.connectionDistance));
    header.numRepetitions = std::max(1, header.numRepetitions);
    header.numBranches = header.getNumBranches();
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
    cell.energy = std::min(std::max(cell.energy, 50.0f), 250.0f);

    switch (cell.getCellFunctionType()) {
    case CellFunction_Constructor: {
        auto& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
        if (constructor.mode < 0) {
            constructor.mode = 0;
        }
        constructor.constructionActivationTime = ((constructor.constructionActivationTime % MaxActivationTime) + MaxActivationTime) % MaxActivationTime;
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
    genome.header.angleAlignment = shapeGenerator->getConstructorAngleAlignment();
    for (auto& cell : genome.cells) {
        auto shapeGenerationResult = shapeGenerator->generateNextConstructionData();
        cell.referenceAngle = shapeGenerationResult.angle;
        cell.numRequiredAdditionalConnections = shapeGenerationResult.numRequiredAdditionalConnections;
    }
}

void _GenomeEditorWindow::setCurrentGenome(GenomeDescription const& genome)
{
    _tabDatas.at(_selectedTabIndex).genome = genome;
}

