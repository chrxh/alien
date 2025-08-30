#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"

#include "EngineInterface/GenomeDescriptionInfoService.h"

#include "PersisterInterface/SerializerService.h"

#include "AlienGui.h"
#include "ChangeColorDialog.h"
#include "GenomeWindowEditData.h"
#include "GenomeTabLayoutData.h"
#include "GenomeTabWidget.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include <ImFileDialog.h>

void GenomeEditorWindow::openTab(std::optional<uint64_t> const& creatureId, GenomeDescription const& genome, bool openEditorIfClosed)
{
    if (openEditorIfClosed) {
        setOn(false);
        delayedExecution([this] { setOn(true); });
    }
    if (_tabs.size() == 1 && _tabs.front()->isEmpty() && _tabs.front()->isDraft()) {
        _tabs.clear();
    }
    std::optional<int> tabIndex;
    if (creatureId.has_value()) {
        for (auto const& [index, tab] : _tabs | boost::adaptors::indexed(0)) {
            if (!tab->isDraft() && tab->getCreatureId() == creatureId) {
                tabIndex = toInt(index);
            }
        }
    }
    if (tabIndex) {
        _tabIndexToSelect = *tabIndex;
        _tabs.at(*tabIndex)->resetChanges();
    } else {
        if (creatureId.has_value()) {
            onScheduleAddCreatureTab(creatureId.value(), genome);
        } else {
            onScheduleAddDraftTab(genome);
        }
    }
}

GenomeDescription GenomeEditorWindow::getCurrentCreature() const
{
    return GenomeDescription();
}

GenomeEditorWindow::GenomeEditorWindow()
    : AlienWindow("Genome editor", "windows.genome editor", false, true, {500.0f, 300.0f})
{
}

void GenomeEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    ChangeColorDialog::get().setup();

    _simulationFacade = simulationFacade;
    _genomeEditData = std::make_shared<_GenomeWindowEditData>();

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getValue("windows.genome editor.starting path", path.string());

    // Initialize the first tab with a draft creature
    _tabs.emplace_back(_GenomeTabWidget::createDraftTab(_simulationFacade, _genomeEditData, getDefaultGenome()));
}

void GenomeEditorWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.genome editor.starting path", _startingPath);
}

void GenomeEditorWindow::processIntern()
{
    processToolbar();
    processTabWidget();
}

bool GenomeEditorWindow::isShown()
{
    return _on && EditorController::get().isOn();
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
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_PALETTE).tooltip("Change the color of all nodes with a certain color"))) {
        ChangeColorDialog::get().open(_tabs.at(_selectedTabIndex)->getEditData());
    }

    ImGui::SameLine();
    AlienGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters()
                                    .text(ICON_FA_SYRINGE)
                                    .tooltip("Inject the current genome to the creature in the simulation")
                                    .disabled(!_tabs.at(_selectedTabIndex)->hasCreaturesGenomeBeChanged()))) {
        onInjectGenome();
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_EGG).tooltip("Create a seed with current genome"))) {
        onCreateSeed();
    }

    AlienGui::Separator();
}

void GenomeEditorWindow::processTabWidget()
{
    if (ImGui::BeginChild("TabWidget", ImVec2(0, 0), 0, 0)) {

        if (ImGui::BeginTabBar(
                "##CreatureTabWidget", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown | ImGuiTabBarFlags_Reorderable)) {

            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                onScheduleAddDraftTab(getDefaultGenome());
            }
            AlienGui::Tooltip("New creature");

            std::optional<int> tabIndexToSelect = _tabIndexToSelect;
            std::optional<int> tabToDelete;
            _tabIndexToSelect.reset();

            // Process tabs
            for (auto const& [index, creatureTab] : _tabs | boost::adaptors::indexed(0)) {

                bool open = true;
                bool* openPtr = nullptr;
                if (_tabs.size() > 1) {
                    openPtr = &open;
                }
                int flags = (tabIndexToSelect && *tabIndexToSelect == index) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;

                pushStyleColorForTab(creatureTab);
                if (ImGui::BeginTabItem((creatureTab->getName() + "###" + std::to_string(creatureTab->getTabId())).c_str(), openPtr, flags)) {
                    _selectedTabIndex = toInt(index);
                    creatureTab->process();
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
    GenericFileDialog::get().showOpenFileDialog("Open genome", "Genome (*.genome){.genome},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        GenomeDescription genome;
        if (!SerializerService::get().deserializeGenomeFromFile(genome, firstFilename.string())) {
            GenericMessageDialog::get().information("Open genome", "The selected file could not be opened.");
        } else {
            openTab(std::nullopt, genome, false);
        }
    });
}

void GenomeEditorWindow::onSaveGenome()
{
    GenericFileDialog::get().showSaveFileDialog("Save genome", "Genome (*.genome){.genome},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        auto const& selectedTab = _tabs.at(_selectedTabIndex);
        auto genome = selectedTab->getGenomeDescription();
        if (!SerializerService::get().serializeGenomeToFile(firstFilename.string(), genome)) {
            GenericMessageDialog::get().information("Save genome", "The selected file could not be saved.");
        }
    });
}

void GenomeEditorWindow::onInjectGenome()
{
    auto const& tab = _tabs.at(_selectedTabIndex);
    auto success = _simulationFacade->changeCreature(tab->getCreatureId(), tab->getGenomeDescription());
    tab->onGenomeIntoCreatureInjected();
    if (success) {
        printOverlayMessage("Genome injected");
    } else {
        GenericMessageDialog::get().information("Error", "The genome could not be injected since the creature no longer exists.");
        tab->convertToDraftTab();
    }
}

void GenomeEditorWindow::onCreateSeed()
{
    auto pos = Viewport::get().getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto tab = _tabs.at(_selectedTabIndex);
    auto genome = tab->getGenomeDescription();

    auto parameter = _simulationFacade->getSimulationParameters();
    auto numNodes = GenomeDescriptionInfoService::get().getNumberOfNodes(genome);
    auto energy = parameter.normalCellEnergy.value[EditorModel::get().getDefaultColorCode()] * toFloat(numNodes * 2 + 1);
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .cells({
                CellDescription()
                    .pos(pos)
                    .energy(energy)
                    .stiffness(1.0f)
                    .color(EditorModel::get().getDefaultColorCode())
                    .cellTypeData(ConstructorDescription().geneIndex(0)),
            })
            .genome(genome),
    });
    _simulationFacade->addAndSelectSimulationData(std::move(data));
    EditorModel::get().update();

    printOverlayMessage("Seed created");
}

void GenomeEditorWindow::onScheduleAddCreatureTab(uint64_t creatureId, GenomeDescription const& genome)
{
    auto const& currentTab = _tabs.at(_selectedTabIndex);
    _tabToAdd = _GenomeTabWidget::createCreatureTab(_simulationFacade, _genomeEditData, creatureId, genome, currentTab->getLayoutData()->clone());
}

void GenomeEditorWindow::onScheduleAddDraftTab(GenomeDescription const& genome)
{
    auto const& currentTab = _tabs.at(_selectedTabIndex);
    _tabToAdd = _GenomeTabWidget::createDraftTab(_simulationFacade, _genomeEditData, genome, currentTab->getLayoutData()->clone());
}

void GenomeEditorWindow::pushStyleColorForTab(GenomeTabWidget const& creatureTab)
{
    if (creatureTab->isDraft()) {

        // Use default colors
        auto const& style = ImGui::GetStyle();
        ImGui::PushStyleColor(ImGuiCol_Tab, style.Colors[ImGuiCol_Tab]);
        ImGui::PushStyleColor(ImGuiCol_TabActive, style.Colors[ImGuiCol_TabActive]);
        ImGui::PushStyleColor(ImGuiCol_TabHovered, style.Colors[ImGuiCol_TabHovered]);
    } else {
        // Use creature ID to create a unique color
        auto creatureId = creatureTab->getTabId();
        auto h = 0.1f + toFloat(creatureId % 20) / 20.0f * 0.5f;
        auto s = 0.4f + toFloat(creatureId % 10) / 10.0f * 0.4f;
        ImGui::PushStyleColor(ImGuiCol_Tab, ImColor::HSV(h, s, 0.35f).Value);
        ImGui::PushStyleColor(ImGuiCol_TabActive, ImColor::HSV(h, s, 0.62f).Value);
        ImGui::PushStyleColor(ImGuiCol_TabHovered, ImColor::HSV(h, s, 0.7f).Value);
    }
}

GenomeDescription GenomeEditorWindow::getDefaultGenome()
{
    return GenomeDescription()
        .name("Draft " + std::to_string(++_sequenceNumberForCreatedGenomes))
        .genes({
            GeneDescription().name("Gene 1").nodes({NodeDescription()}).separation(true),
        });
}
