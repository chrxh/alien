#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <ImFileDialog.h>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/PreviewDescriptionService.h"
#include "PersisterInterface/SerializerService.h"
#include "EngineInterface/ShapeGenerator.h"

#include "AlienImGui.h"
#include "CellTypeStrings.h"
#include "DelayedExecutionController.h"
#include "EditorModel.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "HelpStrings.h"
#include "UploadSimulationDialog.h"
#include "ChangeColorDialog.h"
#include "EditorController.h"

namespace
{
    auto const PreviewHeight = 300.0f;
    auto const ContentHeaderTextWidth = 215.0f;
    auto const ContentTextWidth = 190.0f;
    auto const DynamicTableHeaderColumnWidth = 335.0f;
    auto const DynamicTableColumnWidth = 308.0f;
    auto const SubWindowRightMargin = 0.0f;
}

void GenomeEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    _tabDatas = {TabData()};

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getValue("windows.genome editor.starting path", path.string());
    _previewHeight = GlobalSettings::get().getValue("windows.genome editor.preview height", scale(PreviewHeight));
    ChangeColorDialog::get().setup([&] { return getCurrentGenome(); }, [&](GenomeDescription const& genome) { setCurrentGenome(genome); });
}

void GenomeEditorWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.genome editor.starting path", _startingPath);
    GlobalSettings::get().setValue("windows.genome editor.preview height", _previewHeight);
}

void GenomeEditorWindow::openTab(GenomeDescription const& genome, bool openGenomeEditorIfClosed)
{
    if (openGenomeEditorIfClosed) {
        setOn(false);
        delayedExecution([this] { setOn(true); });
    }
    if (_tabDatas.size() == 1 && _tabDatas.front().genome._cells.empty()) {
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

GenomeDescription const& GenomeEditorWindow::getCurrentGenome() const
{
    return _tabDatas.at(_selectedTabIndex).genome;
}

GenomeEditorWindow::GenomeEditorWindow()
    : AlienWindow("Genome editor", "windows.genome editor", false)
{}

namespace
{
    std::string
    generateShortDescription(int index, CellGenomeDescription const& cell, std::optional<ShapeGeneratorResult> const& shapeGeneratorResult, bool isFirstOrLast)
    {
        auto result = "No. " + std::to_string(index + 1) + ", Type: " + Const::CellTypeToStringMap.at(cell.getCellType())
            + ", Color: " + std::to_string(cell._color);
        if (!isFirstOrLast) {
            auto referenceAngle = shapeGeneratorResult ? shapeGeneratorResult->angle : cell._referenceAngle;
            result += ", Angle: " + StringHelper::format(referenceAngle, 1);
        }
        result += ", Energy: " + StringHelper::format(cell._energy, 1);
        return result;
    }
}

void GenomeEditorWindow::processIntern()
{
    processToolbar();
    processEditor();
}

bool GenomeEditorWindow::isShown()
{
    return _on && EditorController::get().isOn();
}

void GenomeEditorWindow::processToolbar()
{
    if (_tabDatas.empty()) {
        return;
    }
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);

    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN))) {
        onOpenGenome();
    }
    AlienImGui::Tooltip("Open genome from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SAVE))) {
        onSaveGenome();
    }
    AlienImGui::Tooltip("Save genome to file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_UPLOAD))) {
        onUploadGenome();
    }
    AlienImGui::Tooltip("Share your genome with other users:\nYour current genome will be uploaded to the server and made visible in the browser.");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_COPY))) {
        _copiedGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(selectedTab.genome);
        printOverlayMessage("Genome copied");
    }
    AlienImGui::Tooltip("Copy genome");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLUS))) {
        onAddNode();
    }
    AlienImGui::Tooltip("Add cell");

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome._cells.empty());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_MINUS))) {
        onDeleteNode();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete cell");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    auto& selectedNode = selectedTab.selectedNode;
    ImGui::BeginDisabled(!(selectedNode && *selectedNode > 0));
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_CHEVRON_UP))) {
        onNodeDecreaseSequenceNumber();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Decrease sequence number of selected cell");

    ImGui::SameLine();
    ImGui::BeginDisabled(!(selectedNode && *selectedNode < selectedTab.genome._cells.size() - 1));
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_CHEVRON_DOWN))) {
        onNodeIncreaseSequenceNumber();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Increase sequence number of selected cell");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome._cells.empty());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_EXPAND_ARROWS_ALT))) {
        _expandNodes = true;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Expand all cells");

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome._cells.empty());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_COMPRESS_ARROWS_ALT))) {
        _expandNodes = false;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Collapse all cells");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedTab.genome._cells.empty());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PALETTE))) {
        ChangeColorDialog::get().open();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Change the color of all cells with a specific color");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SEEDLING))) {
        onCreateSpore();
    }
    AlienImGui::Tooltip("Create a spore with current genome");

    AlienImGui::Separator();
}

void GenomeEditorWindow::processEditor()
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

void GenomeEditorWindow::processTab(TabData& tab)
{
    _previewHeight = std::min(ImGui::GetContentRegionAvail().y - 10.0f, std::max(10.0f, _previewHeight));
    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - _previewHeight), true)) {
        AlienImGui::Group("General properties");
        processGenomeHeader(tab);

        AlienImGui::Group("Construction sequence");
        processConstructionSequence(tab);
    }
    ImGui::EndChild();

    AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters().additive(false), _previewHeight);

    AlienImGui::Group("Preview (reference configuration)", Const::GenomePreviewTooltip);
    ImGui::SameLine();
    if (ImGui::BeginChild("##child4", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        showPreview(tab);
    }
    ImGui::EndChild();
}

void GenomeEditorWindow::processGenomeHeader(TabData& tab)
{
    AlienImGui::DynamicTableLayout table(DynamicTableHeaderColumnWidth);
    if (table.begin()) {
        std::vector ShapeStrings = {"Custom"s, "Segment"s, "Triangle"s, "Rectangle"s, "Hexagon"s, "Loop"s, "Tube"s, "Lolli"s, "Small lolli"s, "Zigzag"s};
        auto origShape = tab.genome._header._shape;
        if (AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Geometry").values(ShapeStrings).textWidth(ContentHeaderTextWidth).tooltip(Const::GenomeGeometryTooltip),
                tab.genome._header._shape)) {
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
            tab.genome._header._connectionDistance);
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Stiffness")
                .format("%.2f")
                .step(0.05f)
                .textWidth(ContentHeaderTextWidth)
                .tooltip(Const::GenomeStiffnessTooltip),
            tab.genome._header._stiffness);
        if (tab.genome._header._shape == ConstructionShape_Custom) {
            table.next();
            AlienImGui::AngleAlignmentCombo(
                AlienImGui::AngleAlignmentComboParameters()
                    .name("Angle alignment")
                    .textWidth(ContentHeaderTextWidth)
                    .tooltip(Const::GenomeAngleAlignmentTooltip),
                tab.genome._header._angleAlignment);
        }
        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Separation").textWidth(ContentHeaderTextWidth).tooltip(Const::GenomeSeparationConstructionTooltip),
            tab.genome._header._separateConstruction);
        table.next();
        if (!tab.genome._header._separateConstruction) {
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Number of branches").textWidth(ContentHeaderTextWidth).tooltip(Const::GenomeNumBranchesTooltip),
                tab.genome._header._numBranches);
            table.next();
        }
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Repetitions per branch")
                .infinity(true)
                .textWidth(ContentHeaderTextWidth)
                .tooltip(Const::GenomeRepetitionsPerBranchTooltip),
            tab.genome._header._numRepetitions);
        if (tab.genome._header._numRepetitions > 1) {
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Concatenation angle #1")
                    .format("%.1f")
                    .textWidth(ContentHeaderTextWidth)
                    .tooltip(Const::GenomeConcatenationAngle1)
                ,
                tab.genome._header._concatenationAngle1);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Concatenation angle #2")
                    .format("%.1f")
                    .textWidth(ContentHeaderTextWidth)
                    .tooltip(Const::GenomeConcatenationAngle2),
                tab.genome._header._concatenationAngle2);
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Front angle")
                .format("%.1f")
                .textWidth(ContentHeaderTextWidth),
            tab.genome._header._frontAngle);
        table.end();
    }
    validateAndCorrect(tab.genome._header);
}

namespace 
{
    void applyNewCellType(CellGenomeDescription&cell, CellType type)
    {
        switch (type) {
        case CellType_Base: {
            cell._cellTypeData = BaseGenomeDescription();
        } break;
        case CellType_Depot: {
            cell._cellTypeData = DepotGenomeDescription();
        } break;
        case CellType_Constructor: {
            cell._cellTypeData = ConstructorGenomeDescription().genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()));
        } break;
        case CellType_Sensor: {
            cell._cellTypeData = SensorGenomeDescription();
        } break;
        case CellType_Oscillator: {
            cell._cellTypeData = OscillatorGenomeDescription();
        } break;
        case CellType_Attacker: {
            cell._cellTypeData = AttackerGenomeDescription();
        } break;
        case CellType_Injector: {
            cell._cellTypeData = InjectorGenomeDescription().genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()));
        } break;
        case CellType_Muscle: {
            cell._cellTypeData = MuscleGenomeDescription();
        } break;
        case CellType_Defender: {
            cell._cellTypeData = DefenderGenomeDescription();
        } break;
        case CellType_Reconnector: {
            cell._cellTypeData = ReconnectorGenomeDescription();
        } break;
        case CellType_Detonator: {
            cell._cellTypeData = DetonatorGenomeDescription();
        } break;
        }
    }
}

void GenomeEditorWindow::processConstructionSequence(TabData& tab)
{
    int index = 0;

    auto shapeGenerator = ShapeGeneratorFactory::create(tab.genome._header._shape);
    for (auto& cell : tab.genome._cells) {
        auto isFirst = index == 0;
        auto isLast = index == tab.genome._cells.size() - 1;
        auto isFirstOrLast = isFirst || isLast;
        std::optional<ShapeGeneratorResult> shapeGeneratorResult =
            shapeGenerator ? std::make_optional(shapeGenerator->generateNextConstructionData()) : std::nullopt;

        ImGui::PushID(index);

        float h, s, v;
        AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(h, s * 0.5f, v));
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_Framed;

        auto isNodeSelected = tab.selectedNode && *tab.selectedNode == index;
        if (isNodeSelected) {
            flags |= ImGuiTreeNodeFlags_Selected;
        } else {
            ImGui::PushStyleColor(ImGuiCol_Header, static_cast<ImVec4>(ImColor::HSV(0, 0, 0, 0)));
        }
        if (_nodeIndexToJump) {
            if (_nodeIndexToJump == index) {
                ImGui::SetScrollHereY();
                ImGui::SetNextItemOpen(true);
            } else {
                ImGui::SetNextItemOpen(false);
            }
        }

        if (_expandNodes) {
            ImGui::SetNextItemOpen(*_expandNodes);
        }
        ImGui::PushFont(StyleRepository::get().getSmallBoldFont());
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
    _nodeIndexToJump.reset();
}

void GenomeEditorWindow::processNode(
    TabData& tab,
    CellGenomeDescription& cell,
    std::optional<ShapeGeneratorResult> const& shapeGeneratorResult,
    bool isFirst,
    bool isLast)
{
    auto type = cell.getCellType();
    auto isFirstOrLast = isFirst || isLast;

    AlienImGui::DynamicTableLayout table(DynamicTableColumnWidth);
    if (table.begin()) {
        if (AlienImGui::CellTypeCombo(
                AlienImGui::CellTypeComboParameters()
                    .name("Function")
                    .textWidth(ContentTextWidth)
                    .includeStructureAndFreeCells(false)
                    .tooltip(Const::getCellTypeTooltip(type)),
                type)) {
            applyNewCellType(cell, type);
        }
        table.next();
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(ContentTextWidth).tooltip(Const::GenomeColorTooltip), cell._color);
        if (!isFirstOrLast) {
            table.next();
            auto referenceAngle = shapeGeneratorResult ? shapeGeneratorResult->angle : cell._referenceAngle;
            if (AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Angle").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::GenomeAngleTooltip), referenceAngle)) {
                updateGeometry(tab.genome, tab.genome._header._shape);
                tab.genome._header._shape = ConstructionShape_Custom;
            }
            cell._referenceAngle = referenceAngle;
        }
        table.next();
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(ContentTextWidth).format("%.1f").tooltip(Const::GenomeEnergyTooltip), cell._energy);
        if (!isFirst) {
            table.next();
            auto numRequiredAdditionalConnections =
                shapeGeneratorResult ? shapeGeneratorResult->numRequiredAdditionalConnections : cell._numRequiredAdditionalConnections;
            numRequiredAdditionalConnections = numRequiredAdditionalConnections + 1;
            if (AlienImGui::InputInt(
                    AlienImGui::InputIntParameters()
                        .name("Required connections")
                        .textWidth(ContentTextWidth)
                        .tooltip(Const::GenomeRequiredConnectionsTooltip),
                    numRequiredAdditionalConnections)) {
                updateGeometry(tab.genome, tab.genome._header._shape);
                tab.genome._header._shape = ConstructionShape_Custom;
            }
            cell._numRequiredAdditionalConnections = numRequiredAdditionalConnections - 1;
        }

        table.next();
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Signal routing restriction").textWidth(ContentTextWidth),
            cell._signalRoutingRestriction._active);
        if (cell._signalRoutingRestriction._active) {
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Signal base angle").format("%.1f").step(0.5f).textWidth(ContentTextWidth),
                cell._signalRoutingRestriction._baseAngle);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Signal opening angle").format("%.1f").step(0.5f).textWidth(ContentTextWidth),
                cell._signalRoutingRestriction._openingAngle);
        }

        switch (type) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            auto& transmitter = std::get<DepotGenomeDescription>(cell._cellTypeData);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Energy distribution")
                    .values({"Connected cells", "Transmitters and constructors"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeTransmitterEnergyDistributionTooltip),
                transmitter._mode);
        } break;
        case CellType_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(cell._cellTypeData);

            int autoTriggerInterval = constructor._autoTriggerInterval == 0 ? 0 : 1;
            table.next();
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Activation mode")
                        .textWidth(ContentTextWidth)
                        .values({"Manual", "Automatic"})
                        .tooltip(Const::GenomeConstructorActivationModeTooltip),
                    autoTriggerInterval)) {
                constructor._autoTriggerInterval = autoTriggerInterval;

            }
            if (autoTriggerInterval == 1) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Interval").textWidth(ContentTextWidth).tooltip(Const::GenomeConstructorIntervalTooltip), constructor._autoTriggerInterval);
                constructor._autoTriggerInterval = std::max(1, constructor._autoTriggerInterval);
            }
            table.next();
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters()
                    .name("Offspring activation time")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeConstructorOffspringActivationTime),
                constructor._constructionActivationTime);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #1")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeConstructorAngle1Tooltip),
                constructor._constructionAngle1);
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Construction angle #2")
                    .format("%.1f")
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeConstructorAngle2Tooltip),
                constructor._constructionAngle2);
        } break;
        case CellType_Sensor: {
            auto& sensor = std::get<SensorGenomeDescription>(cell._cellTypeData);

            int autoTriggerInterval = sensor._autoTriggerInterval == 0 ? 0 : 1;
            table.next();
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Activation mode")
                        .textWidth(ContentTextWidth)
                        .values({"Manual", "Automatic"})
                        .tooltip(Const::GenomeConstructorActivationModeTooltip),
                    autoTriggerInterval)) {
                sensor._autoTriggerInterval = autoTriggerInterval;
            }
            if (autoTriggerInterval == 1) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Interval").textWidth(ContentTextWidth).tooltip(Const::GenomeConstructorIntervalTooltip),
                    sensor._autoTriggerInterval);
                sensor._autoTriggerInterval = std::max(1, sensor._autoTriggerInterval);
            }

            table.next();
            AlienImGui::ComboOptionalColor(
                AlienImGui::ComboColorParameters().name("Scan color").textWidth(ContentTextWidth).tooltip(Const::GenomeSensorScanColorTooltip), sensor._restrictToColor);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Scan mutants")
                    .values({"None", "Same mutants", "Other mutants", "Free cells", "Handcrafted cells", "Less complex mutants", "More complex mutants"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::SensorRestrictToMutantsTooltip),
                sensor._restrictToMutants);


            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Min density")
                    .format("%.2f")
                    .step(0.05f)
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeSensorMinDensityTooltip),
                sensor._minDensity);
            table.next();
            AlienImGui::InputOptionalInt(
                AlienImGui::InputIntParameters().name("Min range").textWidth(ContentTextWidth).tooltip(Const::GenomeSensorMinRangeTooltip), sensor._minRange);
            table.next();
            AlienImGui::InputOptionalInt(
                AlienImGui::InputIntParameters().name("Max range").textWidth(ContentTextWidth).tooltip(Const::GenomeSensorMaxRangeTooltip), sensor._maxRange);
        } break;
        case CellType_Oscillator: {
            auto& oscillator = std::get<OscillatorGenomeDescription>(cell._cellTypeData);
            table.next();
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Pulse interval").textWidth(ContentTextWidth).tooltip(Const::GenomeOscillatorPulseIntervalTooltip),
                oscillator._autoTriggerInterval);
            bool alternation = oscillator._alternationInterval > 0;
            table.next();
            if (AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(ContentTextWidth).tooltip(Const::GenomeOscillatorAlternatingPulsesTooltip),
                    alternation)) {
                oscillator._alternationInterval = alternation ? 1 : 0;
            }
            if (alternation) {
                table.next();
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Pulses per phase").textWidth(ContentTextWidth).tooltip(Const::GenomeOscillatorPulsesPerPhaseTooltip),
                    oscillator._alternationInterval);
            }
        } break;
        case CellType_Attacker: {
            //auto& attacker = std::get<AttackerGenomeDescription>(cell._cellTypeData);
        } break;
        case CellType_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(cell._cellTypeData);

            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .textWidth(ContentTextWidth)
                    .values({"Only empty cells", "All cells"})
                    .tooltip(Const::GenomeInjectorModeTooltip),
                injector._mode);
        } break;
        case CellType_Muscle: {
            auto& muscle = std::get<MuscleGenomeDescription>(cell._cellTypeData);
            table.next();
            auto mode = toInt(muscle.getMode());
            if (AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                        .values({"Auto bending", "Manual bending", "Angle bending", "Auto crawling", "Manual crawling", "Direct movement"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeMuscleModeTooltip),
                mode)) {
                if (mode == MuscleMode_AutoBending) {
                    muscle.mode(AutoBendingGenomeDescription());
                } else if (mode == MuscleMode_ManualBending) {
                    muscle.mode(ManualBendingGenomeDescription());
                } else if (mode == MuscleMode_AngleBending) {
                    muscle.mode(AngleBendingGenomeDescription());
                } else if (mode == MuscleMode_AutoCrawling) {
                    muscle.mode(AutoCrawlingGenomeDescription());
                } else if (mode == MuscleMode_ManualCrawling) {
                    muscle.mode(ManualCrawlingGenomeDescription());
                } else if (mode == MuscleMode_DirectMovement) {
                    muscle.mode(DirectMovementGenomeDescription());
                }
            }
            if (mode == MuscleMode_AutoBending) {
                auto& bending = std::get<AutoBendingGenomeDescription>(muscle._mode);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters()
                        .name("Max angle deviation")
                        .format("%.2f")
                        .textWidth(ContentHeaderTextWidth),
                    bending._maxAngleDeviation);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    bending._frontBackVelRatio);
            } else if (mode == MuscleMode_ManualBending) {
                auto& bending = std::get<ManualBendingGenomeDescription>(muscle._mode);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    bending._maxAngleDeviation);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    bending._frontBackVelRatio);
            } else if (mode == MuscleMode_AngleBending) {
                auto& bending = std::get<AngleBendingGenomeDescription>(muscle._mode);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    bending._maxAngleDeviation);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    bending._frontBackVelRatio);
            } else if (mode == MuscleMode_AutoCrawling) {
                auto& crawling = std::get<AutoCrawlingGenomeDescription>(muscle._mode);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Max distance deviation").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    crawling._maxDistanceDeviation);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    crawling._frontBackVelRatio);
            } else if (mode == MuscleMode_ManualCrawling) {
                auto& crawling = std::get<ManualCrawlingGenomeDescription>(muscle._mode);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Max distance deviation").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    crawling._maxDistanceDeviation);
                table.next();
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Front back ratio").format("%.2f").step(0.05f).textWidth(ContentHeaderTextWidth),
                    crawling._frontBackVelRatio);
            }
        } break;
        case CellType_Defender: {
            auto& defender = std::get<DefenderGenomeDescription>(cell._cellTypeData);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Mode")
                    .values({"Anti-attacker", "Anti-injector"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::GenomeDefenderModeTooltip),
                defender._mode);
        } break;
        case CellType_Reconnector: {
            auto& reconnector = std::get<ReconnectorGenomeDescription>(cell._cellTypeData);
            table.next();
            AlienImGui::ComboOptionalColor(
                AlienImGui::ComboColorParameters().name("Restrict to color").textWidth(ContentTextWidth).tooltip(Const::GenomeReconnectorRestrictToColorTooltip),
                reconnector._restrictToColor);
            table.next();
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Restrict to mutants")
                    .values({"None", "Same mutants", "Other mutants", "Free cells", "Handcrafted cells", "Less complex mutants", "More complex mutants"})
                    .textWidth(ContentTextWidth)
                    .tooltip(Const::ReconnectorRestrictToMutantsTooltip),
                reconnector._restrictToMutants);
        } break;
        case CellType_Detonator: {
            table.next();
            auto& detonator = std::get<DetonatorGenomeDescription>(cell._cellTypeData);
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Countdown").textWidth(ContentTextWidth).tooltip(Const::GenomeDetonatorCountdownTooltip),
                detonator._countdown);
            detonator._countdown = std::min(0xffff, std::max(0, detonator._countdown));
        } break;
        }

        table.end();

        if (ImGui::TreeNodeEx("Neural network", ImGuiTreeNodeFlags_None)) {
            AlienImGui::NeuronSelection(
                AlienImGui::NeuronSelectionParameters().rightMargin(SubWindowRightMargin),
                cell._neuralNetwork._weights,
                cell._neuralNetwork._biases,
                cell._neuralNetwork._activationFunctions);
            ImGui::TreePop();
        }

        switch (type) {
        case CellType_Base: {
        } break;
        case CellType_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(cell._cellTypeData);
            processSubGenomeWidgets(tab, constructor);
        } break;
        case CellType_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(cell._cellTypeData);
            processSubGenomeWidgets(tab, injector);
        } break;
        }
    }
    validateAndCorrect(cell);
}

template <typename Description>
void GenomeEditorWindow::processSubGenomeWidgets(TabData const& tab, Description& desc)
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
            desc.genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()));
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Copy")) {
            _copiedGenome = desc.isMakeGenomeCopy() ? GenomeDescriptionConverterService::get().convertDescriptionToBytes(tab.genome) : desc.getGenomeData();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!_copiedGenome.has_value());
        if (AlienImGui::Button("Paste")) {
            desc._genome = *_copiedGenome;
            printOverlayMessage("Genome pasted");
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (AlienImGui::Button("Edit")) {
            auto genomeToOpen = desc.isMakeGenomeCopy()
                ? tab.genome
                : GenomeDescriptionConverterService::get().convertBytesToDescription(desc.getGenomeData());
            openTab(genomeToOpen, false);
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Set self-copy")) {
            desc.makeSelfCopy();
        }
    }
    ImGui::EndChild();
}


void GenomeEditorWindow::onOpenGenome()
{
    GenericFileDialog::get().showOpenFileDialog("Open genome", "Genome (*.genome){.genome},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        std::vector<uint8_t> genomeData;
        if (!SerializerService::get().deserializeGenomeFromFile(genomeData, firstFilename.string())) {
            GenericMessageDialog::get().information("Open genome", "The selected file could not be opened.");
        } else {
            openTab(GenomeDescriptionConverterService::get().convertBytesToDescription(genomeData), false);
        }
    });
}

void GenomeEditorWindow::onSaveGenome()
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save genome", "Genome (*.genome){.genome},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            auto const& selectedTab = _tabDatas.at(_selectedTabIndex);
            auto genomeData = GenomeDescriptionConverterService::get().convertDescriptionToBytes(selectedTab.genome);
            if (!SerializerService::get().serializeGenomeToFile(firstFilename.string(), genomeData)) {
                GenericMessageDialog::get().information("Save genome", "The selected file could not be saved.");
            }
        });
}

void GenomeEditorWindow::onUploadGenome()
{
    UploadSimulationDialog::get().open(NetworkResourceType_Genome);
}

void GenomeEditorWindow::onAddNode()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);
    auto& cells = tabData.genome._cells;

    CellGenomeDescription newNode;
    if (tabData.selectedNode) {
        newNode._color = cells.at(*tabData.selectedNode)._color;
        cells.insert(cells.begin() + *tabData.selectedNode + 1, newNode);
        ++(*tabData.selectedNode);
    } else {
        if (!cells.empty()) {
            newNode._color = cells.back()._color;
        }
        cells.emplace_back(newNode);
        tabData.selectedNode = toInt(cells.size() - 1);
    }
}

void GenomeEditorWindow::onDeleteNode()
{
    auto& tabData = _tabDatas.at(_selectedTabIndex);
    auto& cells = tabData.genome._cells;

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

void GenomeEditorWindow::onNodeDecreaseSequenceNumber()
{
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selectedNode = selectedTab.selectedNode;
    std::swap(selectedTab.genome._cells.at(*selectedNode), selectedTab.genome._cells.at(*selectedNode - 1));
    --(*selectedNode);
    _expandNodes = false;
}

void GenomeEditorWindow::onNodeIncreaseSequenceNumber()
{
    auto& selectedTab = _tabDatas.at(_selectedTabIndex);
    auto& selectedNode = selectedTab.selectedNode;
    std::swap(selectedTab.genome._cells.at(*selectedNode), selectedTab.genome._cells.at(*selectedNode + 1));
    ++(*selectedNode);
    _expandNodes = false;
}

void GenomeEditorWindow::onCreateSpore()
{
    auto pos = Viewport::get().getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto genomeDesc = getCurrentGenome();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(genomeDesc);

    auto parameter = _simulationFacade->getSimulationParameters();
    auto numNodes = GenomeDescriptionConverterService::get().getNumNodesRecursively(genome, true);
    auto energy = parameter.normalCellEnergy.value[EditorModel::get().getDefaultColorCode()] * toFloat(numNodes * 2 + 1);
    auto cell = CellDescription()
                    .pos(pos)
                    .energy(energy)
                    .stiffness(1.0f)
                    .color(EditorModel::get().getDefaultColorCode())
                    .cellTypeData(ConstructorDescription().genome(genome));
    auto data = CollectionDescription().addCell(cell);
    _simulationFacade->addAndSelectSimulationData(data);
    EditorModel::get().update();

    printOverlayMessage("Spore created");
}

void GenomeEditorWindow::showPreview(TabData& tab)
{
    auto const& genome = _tabDatas.at(_selectedTabIndex).genome;
    auto preview = PreviewDescriptionService::get().convert(genome, tab.selectedNode, _simulationFacade->getSimulationParameters());
    if (AlienImGui::ShowPreviewDescription(preview, tab.previewZoom, tab.selectedNode)) {
        _nodeIndexToJump = tab.selectedNode;
    }
}

void GenomeEditorWindow::validateAndCorrect(GenomeHeaderDescription& header) const
{
    header._stiffness = std::max(0.0f, std::min(1.0f, header._stiffness));
    header._connectionDistance = std::max(0.5f, std::min(1.5f, header._connectionDistance));
    header._numRepetitions = std::max(1, header._numRepetitions);
    header._numBranches = header.getNumBranches();
}

void GenomeEditorWindow::validateAndCorrect(CellGenomeDescription& cell) const
{
    cell._color = (cell._color + MAX_COLORS) % MAX_COLORS;
    cell._numRequiredAdditionalConnections = (cell._numRequiredAdditionalConnections + MAX_CELL_BONDS) % MAX_CELL_BONDS;
    cell._energy = std::min(std::max(cell._energy, 50.0f), 250.0f);

    switch (cell.getCellType()) {
    case CellType_Constructor: {
        auto& constructor = std::get<ConstructorGenomeDescription>(cell._cellTypeData);
        constructor._autoTriggerInterval = std::min(255, std::max(0, constructor._autoTriggerInterval));
        constructor._constructionActivationTime = ((constructor._constructionActivationTime % MAX_ACTIVATION_TIME) + MAX_ACTIVATION_TIME) % MAX_ACTIVATION_TIME;
    } break;
    case CellType_Sensor: {
        auto& sensor = std::get<SensorGenomeDescription>(cell._cellTypeData);
        sensor._autoTriggerInterval = std::min(255, std::max(0, sensor._autoTriggerInterval));
        sensor._minDensity = std::max(0.0f, std::min(1.0f, sensor._minDensity));
        if (sensor._minRange) {
            sensor._minRange = std::max(0, std::min(127, *sensor._minRange));
        }
        if (sensor._maxRange) {
            sensor._maxRange = std::max(0, std::min(127, *sensor._maxRange));
        }
    } break;
    case CellType_Oscillator: {
        auto& oscillator = std::get<OscillatorGenomeDescription>(cell._cellTypeData);
        oscillator._autoTriggerInterval = std::max(0, oscillator._autoTriggerInterval);
        oscillator._alternationInterval = std::max(0, oscillator._alternationInterval);
    } break;
    }
}

void GenomeEditorWindow::scheduleAddTab(GenomeDescription const& genome)
{
    TabData newTab;
    newTab.id = ++_tabSequenceNumber;
    newTab.genome = genome;
    _tabToAdd = newTab;
}

void GenomeEditorWindow::updateGeometry(GenomeDescription& genome, ConstructionShape shape)
{
    auto shapeGenerator = ShapeGeneratorFactory::create(shape);
    if (!shapeGenerator) {
        return;
    }
    genome._header._angleAlignment = shapeGenerator->getConstructorAngleAlignment();
    for (auto& cell : genome._cells) {
        auto shapeGenerationResult = shapeGenerator->generateNextConstructionData();
        cell._referenceAngle = shapeGenerationResult.angle;
        cell._numRequiredAdditionalConnections = shapeGenerationResult.numRequiredAdditionalConnections;
    }
}

void GenomeEditorWindow::setCurrentGenome(GenomeDescription const& genome)
{
    _tabDatas.at(_selectedTabIndex).genome = genome;
}

