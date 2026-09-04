#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>
#include <ImFileDialog.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDescInfoService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/ObjectColoring.h>
#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/SerializerService.h>

#include "AlienGui.h"
#include "ChangeColorDialog.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "FileTransferController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "GenomeTabLayoutData.h"
#include "GenomeTabWidget.h"
#include "GenomeWindowEditData.h"
#include "OverlayController.h"

namespace
{
    auto constexpr UnsavedChangesText = "Unsaved changes";
    auto constexpr UnsavedChangesChipPadding = 34.0f;

    std::optional<ImColor> getLineageMarkerColor(std::optional<int> const& lineageId)
    {
        if (!lineageId.has_value()) {
            return std::nullopt;
        }
        auto rgb = ObjectColoring::getColorFromId(toUInt32(lineageId.value()));
        return ImColor(toInt((rgb >> 16) & 0xff), toInt((rgb >> 8) & 0xff), toInt(rgb & 0xff));
    }
}

void GenomeEditorWindow::openTab(GenomeDesc const& genome, bool forceNewTab, bool openEditorIfClosed, std::optional<int> lineageId)
{
    if (openEditorIfClosed) {
        setOn(false);
        delayedExecution([this] { setOn(true); });
    }
    if (_tabs.size() == 1 && _tabs.front()->isEmpty()) {
        _tabs.clear();
    }
    auto normalizedGenome = _GenomeTabWidget::normalizeForEditor(genome);
    std::optional<int> tabIndex;
    if (!forceNewTab) {
        for (auto const& [index, tab] : _tabs | boost::adaptors::indexed(0)) {
            auto tabGenome = tab->getGenomeDesc();
            if (normalizedGenome.equalWithoutId(tabGenome)) {
                tabIndex = toInt(index);
            }
        }
    }
    if (tabIndex) {
        _tabIndexToSelect = *tabIndex;
        _tabs.at(*tabIndex)->resetOriginal();
        _tabs.at(*tabIndex)->setLineageId(lineageId);
    } else {
        onScheduleAddTab(normalizedGenome, lineageId);
    }
}

GenomeDesc GenomeEditorWindow::getCurrentGenome() const
{
    return _tabs.at(_selectedTabIndex)->getGenomeDesc();
}

GenomeEditorWindow::GenomeEditorWindow()
    : AlienWindow("Genome editor", "windows.genome editor", false, true, {345.0f, 192.0f}, {1100.0f, 732.0f}, {500.0f, 300.0f})
{}

void GenomeEditorWindow::initIntern()
{
    ChangeColorDialog::get().setup();

    _genomeEditData = std::make_shared<_GenomeWindowEditData>();
    _genomeEditData->showNodeIndex = GlobalSettings::get().getValue(_settingsNode + ".show node index", true);

    // Initialize the first tab with default genome
    _tabs.emplace_back(_GenomeTabWidget::create(_genomeEditData, getDefaultGenome()));
}

void GenomeEditorWindow::shutdownIntern()
{
    GlobalSettings::get().setValue(_settingsNode + ".show node index", _genomeEditData->showNodeIndex);
}

void GenomeEditorWindow::processIntern()
{
    processToolbar();
    processTabWidget();
}

bool GenomeEditorWindow::isShown()
{
    return _on;
}

namespace
{
    float calcUnsavedChangesChipWidth()
    {
        return scaleInverse(ImGui::CalcTextSize(UnsavedChangesText).x) + UnsavedChangesChipPadding;
    }

    void processUnsavedChangesChip()
    {
        AlienGui::Chip(AlienGui::ChipParameters()
                           .text(UnsavedChangesText)
                           .textColor(Const::UnsavedChangesColor)
                           .backgroundColor(Const::UnsavedChangesBackgroundColor)
                           .dotColor(Const::UnsavedChangesColor));
    }
}

void GenomeEditorWindow::processToolbar()
{
    auto hasGenomeChanged = _tabs.at(_selectedTabIndex)->hasGenomeChanged();
    auto creaturesSelected = EditorModel::get().getSelectionShallowData().numCreatures > 0;

    auto toolbarParameters = AlienGui::ToolbarParameters().id("GenomeEditor");
    if (hasGenomeChanged) {
        toolbarParameters.trailing([] { processUnsavedChangesChip(); }).trailingWidth(calcUnsavedChangesChipWidth());
    }

    AlienGui::Toolbar(
        toolbarParameters,
        {AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_FOLDER_OPEN).name("Open genome").tooltip("Open genome from file").action([&] { onOpenGenome(); })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_SAVE).name("Save genome").tooltip("Save genome to file").action([&] { onSaveGenome(); })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters()
                 .icon(ICON_FA_UPLOAD)
                 .name("Share genome")
                 .tooltip("Share your genome with other users:\nYour current genome will be uploaded to the server and made visible in the browser.")),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(ICON_FA_CLONE).name("Clone genome").action([&] { onCloneGenome(); })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_COPY).name("Copy genome").tooltip("Copy genome to clipboard").action([&] { onCopyGenome(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_PASTE)
                                                 .name("Paste genome")
                                                 .tooltip("Paste genome from clipboard")
                                                 .disabled(!_copiedGenome.has_value())
                                                 .action([&] { onPasteGenome(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_WINDOW_CLOSE)
                                                 .name("Close other tabs")
                                                 .tooltip("Close all tabs except the current one")
                                                 .disabled(_tabs.size() <= 1)
                                                 .action([&] { onCloseOtherTabs(); })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_CAMERA)
                                                 .name("Create save point")
                                                 .tooltip("Create save point in this tab")
                                                 .disabled(!hasGenomeChanged)
                                                 .action([&] { onSavepointGenome(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_UNDO)
                                                 .name("Revert to save point")
                                                 .tooltip("Revert genome to save point")
                                                 .disabled(!hasGenomeChanged)
                                                 .action([&] { _tabs.at(_selectedTabIndex)->revertChanges(); })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_PALETTE)
                                                 .name("Change colors")
                                                 .tooltip("Change the color of all nodes with a certain color")
                                                 .action([&] { ChangeColorDialog::get().open(_tabs.at(_selectedTabIndex)->getEditData()); })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_SYRINGE)
                                                 .name("Inject genome")
                                                 .tooltip("Inject the current genome to the selected creatures in the simulation")
                                                 .disabled(!creaturesSelected)
                                                 .action([&] { onInjectGenome(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_SEEDLING)
                                                 .name("Create seed")
                                                 .tooltip("Create a seed with current genome without free energy supply")
                                                 .action([&] { onCreateSeed(false); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_SEEDLING)
                                                 .secondIcon(ICON_FA_BOLT)
                                                 .secondIconOffset({30.0f, 25.0f})
                                                 .name("Create seed with energy")
                                                 .tooltip("Create a seed with current genome with free energy supply")
                                                 .action([&] { onCreateSeed(true); }))});
}

void GenomeEditorWindow::processTabWidget()
{
    if (ImGui::BeginChild("TabWidget", ImVec2(0, 0), 0, 0)) {

        if (ImGui::BeginTabBar(
                "##GenomeTabWidget", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown | ImGuiTabBarFlags_Reorderable)) {

            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                onScheduleAddTab(getDefaultGenome(), std::nullopt);
            }
            AlienGui::Tooltip("New genome");

            std::optional<int> tabIndexToSelect = _tabIndexToSelect;
            std::optional<int> tabToDelete;
            _tabIndexToSelect.reset();

            // Process tabs
            for (auto const& [index, genomeTab] : _tabs | boost::adaptors::indexed(0)) {

                bool open = true;
                bool* openPtr = nullptr;
                if (_tabs.size() > 1) {
                    openPtr = &open;
                }

                if (AlienGui::BeginTabItem(AlienGui::TabItemParameters()
                                               .name(genomeTab->getName())
                                               .id(std::to_string(genomeTab->getTabId()))
                                               .selected(tabIndexToSelect && *tabIndexToSelect == index)
                                               .open(openPtr)
                                               .markerColor(getLineageMarkerColor(genomeTab->getLineageId())))) {
                    _selectedTabIndex = toInt(index);
                    genomeTab->process();
                    AlienGui::EndTabItem();
                }

                if (openPtr && *openPtr == false) {
                    tabToDelete = toInt(index);
                }
            }

            // Delete tab
            if (tabToDelete.has_value()) {
                _tabs.erase(_tabs.begin() + *tabToDelete);
                if (_selectedTabIndex == _tabs.size()) {
                    _selectedTabIndex = toInt(_tabs.size() - 1);
                }
            }

            // Add tabs
            for (auto& tab : _tabsToAdd) {
                _tabs.emplace_back(std::move(tab));
            }
            _tabsToAdd.clear();

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();
}

void GenomeEditorWindow::onOpenGenome()
{
    FileTransferController::get().onOpenGenomeDialog([this](GenomeDesc const& genome) { openTab(genome, true, false); });
}

void GenomeEditorWindow::onSaveGenome()
{
    auto const& selectedTab = _tabs.at(_selectedTabIndex);
    auto genome = selectedTab->getGenomeDesc();
    FileTransferController::get().onSaveGenomeDialog(genome, [selectedTab]() { selectedTab->resetOriginal(); });
}

void GenomeEditorWindow::onCloneGenome()
{
    openTab(getCurrentGenome(), true, false);
}

void GenomeEditorWindow::onCloseOtherTabs()
{
    auto selectedTab = _tabs.at(_selectedTabIndex);
    _tabs.clear();
    _tabs.emplace_back(selectedTab);
    _selectedTabIndex = 0;
}

void GenomeEditorWindow::onCopyGenome()
{
    _copiedGenome = getCurrentGenome();
}

void GenomeEditorWindow::onPasteGenome()
{
    auto const& selectedTab = _tabs.at(_selectedTabIndex);
    selectedTab->setGenomeDesc(_copiedGenome.value());
    selectedTab->resetOriginal();
}

void GenomeEditorWindow::onSavepointGenome()
{
    auto const& selectedTab = _tabs.at(_selectedTabIndex);
    selectedTab->resetOriginal();
}

void GenomeEditorWindow::onInjectGenome()
{
    auto const& selectedTab = _tabs.at(_selectedTabIndex);
    auto numCreatures = _SimulationFacade::get()->injectGenomeToSelectedCreatures(selectedTab->getGenomeDesc());
    printOverlayMessage("Genome injected to " + std::to_string(numCreatures) + (numCreatures == 1 ? " creature" : " creatures"));
    selectedTab->resetOriginal();
}

void GenomeEditorWindow::onCreateSeed(bool provideEnergy)
{
    auto pos = Viewport::get().getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto tab = _tabs.at(_selectedTabIndex);
    auto genome = tab->getGenomeDesc();

    ContentDesc seed;
    seed.addCreature(
        {ObjectDesc()
             .pos(pos)
             .stiffness(1.0f)
             .color(EditorModel::get().getDefaultColorCode())
             .type(CellDesc().headCell(true).constructor(ConstructorDesc()
                                                             .autoTriggerInterval(50)
                                                             .provideEnergy(provideEnergy ? ProvideEnergy_Free : ProvideEnergy_ReduceCellEnergy)
                                                             .geneIndex(0)
                                                             .separation(true)))},
        CreatureDesc(),
        genome);
    DescEditService::get().randomizeLineageIds(seed);

    _SimulationFacade::get()->addAndSelectSimulationData(std::move(seed));
    EditorModel::get().update();

    printOverlayMessage("Seed created");
}

void GenomeEditorWindow::onScheduleAddTab(GenomeDesc const& genome, std::optional<int> lineageId)
{
    auto const& currentTab = _tabs.at(_selectedTabIndex);
    _tabsToAdd.emplace_back(_GenomeTabWidget::create(_genomeEditData, genome, currentTab->getLayoutData()->clone(), lineageId));
}

GenomeDesc GenomeEditorWindow::getDefaultGenome()
{
    return GenomeDesc()
        .name("Draft " + std::to_string(++_sequenceNumberForCreatedGenomes))
        .frontAngle(-180.0f)
        .genes({
            GeneDesc().name("Gene 0").nodes({NodeDesc()}).shape(ConstructorShape_Segment),
        });
}
