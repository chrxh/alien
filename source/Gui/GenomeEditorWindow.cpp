#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <ImFileDialog.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/GenomeDescInfoService.h>
#include <EngineInterface/NumberGenerator.h>
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

void GenomeEditorWindow::openTab(GenomeDesc const& genome, bool forceNewTab, bool openEditorIfClosed)
{
    if (openEditorIfClosed) {
        setOn(false);
        delayedExecution([this] { setOn(true); });
    }
    if (_tabs.size() == 1 && _tabs.front()->isEmpty()) {
        _tabs.clear();
    }
    std::optional<int> tabIndex;
    if (!forceNewTab) {
        for (auto const& [index, tab] : _tabs | boost::adaptors::indexed(0)) {
            auto tabGenome = tab->getGenomeDesc();
            if (genome.equalWithoutId(tabGenome)) {
                tabIndex = toInt(index);
            }
        }
    }
    if (tabIndex) {
        _tabIndexToSelect = *tabIndex;
        _tabs.at(*tabIndex)->resetOriginal();
    } else {
        onScheduleAddTab(genome);
    }
}

GenomeDesc GenomeEditorWindow::getCurrentGenome() const
{
    return _tabs.at(_selectedTabIndex)->getGenomeDesc();
}

GenomeEditorWindow::GenomeEditorWindow()
    : AlienWindow("Genome editor", "windows.genome editor", false, true, {500.0f, 300.0f})
{}

void GenomeEditorWindow::initIntern()
{
    ChangeColorDialog::get().setup();

    _genomeEditData = std::make_shared<_GenomeWindowEditData>();

    // Initialize the first tab with default genome
    _tabs.emplace_back(_GenomeTabWidget::create(_genomeEditData, getDefaultGenome()));
}

void GenomeEditorWindow::shutdownIntern() {}

void GenomeEditorWindow::processIntern()
{
    processToolbar();
    processTabWidget();
}

bool GenomeEditorWindow::isShown()
{
    return _on;
}

void GenomeEditorWindow::processToolbar()
{
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN).tooltip("Open genome from file"))) {
        onOpenGenome();
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_SAVE).tooltip("Save genome to file"))) {
        onSaveGenome();
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(
            AlienGui::ToolbarButtonParameters()
                .text(ICON_FA_UPLOAD)
                .tooltip("Share your genome with other users:\nYour current genome will be uploaded to the server and made visible in the browser."))) {
    }

    ImGui::SameLine();
    AlienGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CLONE).tooltip("Clone genome"))) {
        onCloneGenome();
    }
    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_COPY).tooltip("Copy genome to clipboard"))) {
        onCopyGenome();
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(
            AlienGui::ToolbarButtonParameters().text(ICON_FA_PASTE).tooltip("Paste genome from clipboard").disabled(!_copiedGenome.has_value()))) {
        onPasteGenome();
    }

    ImGui::SameLine();
    auto hasGenomeChanged = _tabs.at(_selectedTabIndex)->hasGenomeChanged();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CAMERA).tooltip("Create save point").disabled(!hasGenomeChanged))) {
        onSavepointGenome();
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(
            AlienGui::ToolbarButtonParameters().text(ICON_FA_UNDO).tooltip("Revert changes on genome").disabled(!hasGenomeChanged))) {
        _tabs.at(_selectedTabIndex)->revertChanges();
    }

    ImGui::SameLine();
    AlienGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_PALETTE).tooltip("Change the color of all nodes with a certain color"))) {
        ChangeColorDialog::get().open(_tabs.at(_selectedTabIndex)->getEditData());
    }

    ImGui::SameLine();
    AlienGui::ToolbarSeparator();

    ImGui::SameLine();
    auto creaturesSelected = EditorModel::get().getSelectionShallowData().numCreatures > 0;
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters()
                                    .text(ICON_FA_SYRINGE)
                                    .tooltip("Inject the current genome to the selected creatures in the simulation")
                                    .disabled(!creaturesSelected))) {
        onInjectGenome();
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(
            AlienGui::ToolbarButtonParameters().text(ICON_FA_SEEDLING).tooltip("Create a seed with current genome without free energy supply"))) {
        onCreateSeed(false);
    }
    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters()
                                    .text(ICON_FA_SEEDLING)
                                    .secondText(ICON_FA_BOLT)
                                    .secondTextOffset({30.0f, 25.0f})
                                    .tooltip("Create a seed with current genome with free energy supply"))) {
        onCreateSeed(true);
    }

    AlienGui::Separator();
}

void GenomeEditorWindow::processTabWidget()
{
    if (ImGui::BeginChild("TabWidget", ImVec2(0, 0), 0, 0)) {

        if (ImGui::BeginTabBar(
                "##GenomeTabWidget", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown | ImGuiTabBarFlags_Reorderable)) {

            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                onScheduleAddTab(getDefaultGenome());
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
                int flags = (tabIndexToSelect && *tabIndexToSelect == index) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;

                pushStyleColorForTab(genomeTab);
                auto tabResult = ImGui::BeginTabItem((genomeTab->getName() + "###" + std::to_string(genomeTab->getTabId())).c_str(), openPtr, flags);
                ImGui::PopStyleColor(3);
                if (tabResult) {
                    _selectedTabIndex = toInt(index);
                    genomeTab->process();
                    ImGui::EndTabItem();
                }
                ImGui::PopStyleColor(3);

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

            // Add tab
            if (_tabToAdd.has_value()) {
                _tabs.emplace_back(_tabToAdd.value());
                _tabToAdd.reset();
            }

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
    //selectedTab->resetOriginal();
    auto genome = selectedTab->getGenomeDesc();
    FileTransferController::get().onSaveGenomeDialog(genome);
}

void GenomeEditorWindow::onCloneGenome()
{
    openTab(getCurrentGenome(), true, false);
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
    auto const& tab = _tabs.at(_selectedTabIndex);
    _SimulationFacade::get()->injectGenomeToSelectedCreatures(tab->getGenomeDesc());
    printOverlayMessage("Genome injected");
}

void GenomeEditorWindow::onCreateSeed(bool provideEnergy)
{
    auto pos = Viewport::get().getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto tab = _tabs.at(_selectedTabIndex);
    auto genome = tab->getGenomeDesc();

    Desc seed;
    seed.addCreature(
        {ObjectDesc()
             .pos(pos)
             .stiffness(1.0f)
             .color(EditorModel::get().getDefaultColorCode())
             .type(
                 CellDesc().constructor(ConstructorDesc().provideEnergy(provideEnergy ? ProvideEnergy_FreeGeneration : ProvideEnergy_CellOnly).geneIndex(0)))},
        CreatureDesc(),
        genome);

    _SimulationFacade::get()->addAndSelectSimulationData(std::move(seed));
    EditorModel::get().update();

    printOverlayMessage("Seed created");
}

void GenomeEditorWindow::onScheduleAddTab(GenomeDesc const& genome)
{
    auto const& currentTab = _tabs.at(_selectedTabIndex);
    _tabToAdd = _GenomeTabWidget::create(_genomeEditData, genome, currentTab->getLayoutData()->clone());
}

void GenomeEditorWindow::pushStyleColorForTab(GenomeTabWidget const& genomeTab)
{
    auto tabId = genomeTab->getTabId();
    auto h = 0.0f + toFloat(tabId % 20) / 20.0f * 1.0f;
    auto s = 0.4f + toFloat(tabId % 10) / 10.0f * 0.6f;
    ImGui::PushStyleColor(ImGuiCol_Tab, ImColor::HSV(h, s, 0.4f).Value);
    ImGui::PushStyleColor(ImGuiCol_TabActive, ImColor::HSV(h, s, 0.7f).Value);
    ImGui::PushStyleColor(ImGuiCol_TabHovered, ImColor::HSV(h, s, 0.8f).Value);
    ImGui::PushStyleColor(ImGuiCol_Button, ImColor::HSV(h, s, 0.8f).Value);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImColor::HSV(h, s, 1.0f).Value);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImColor::HSV(h, s, 0.9f).Value);
}

GenomeDesc GenomeEditorWindow::getDefaultGenome()
{
    return GenomeDesc()
        .name("Draft " + std::to_string(++_sequenceNumberForCreatedGenomes))
        .genes({
            GeneDesc().name("Gene 1").nodes({NodeDesc()}).separation(true).shape(ConstructorShape_Segment),
        });
}
