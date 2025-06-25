#include "CreatureEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/GenomeDescriptionInfoService.h"

#include "AlienGui.h"
#include "CreatureTabLayoutData.h"
#include "CreatureTabWidget.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"

void CreatureEditorWindow::openTab(CreatureDescription const& creature, bool openCreatureEditorIfClosed)
{
    if (openCreatureEditorIfClosed) {
        setOn(false);
        delayedExecution([this] { setOn(true); });
    }
    if (_tabs.size() == 1 && _tabs.front()->isEmpty() && _tabs.front()->isDraft()) {
        _tabs.clear();
    }
    std::optional<int> tabIndex;
    for (auto const& [index, tab] : _tabs | boost::adaptors::indexed(0)) {
        if (!tab->isDraft() && tab->getCreatureId() == creature._id) {
            tabIndex = toInt(index);
        }
    }
    if (tabIndex) {
        //_tabIndexToSelect = *tabIndex;
    } else {
        onScheduleAddTab(creature);
    }
}

CreatureEditorWindow::CreatureEditorWindow()
    : AlienWindow("Creature editor", "windows.creature editor", false, true, {500.0f, 300.0f})
{}

void CreatureEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    // Test genome for the editor
    auto genome = CreatureDescription().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().geneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });

    _tabs.emplace_back(_CreatureTabWidget::createDraftCreatureTab(_simulationFacade, genome));
    //for (int i = 0; i < 10; ++i) {
    //    _tabs.emplace_back(_CreatureTabWidget::createPinnedCreatureTab(GenomeDescription_New(), rand()));
    //}
}

void CreatureEditorWindow::shutdownIntern() {}

void CreatureEditorWindow::processIntern()
{
    processToolbar();
    processTabWidget();
}

bool CreatureEditorWindow::isShown()
{
    return _on && EditorController::get().isOn();
}

void CreatureEditorWindow::processToolbar()
{
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN).tooltip("Open creature from file"))) {
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_SAVE).tooltip("Save creature to file"))) {
    }

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(
            AlienGui::ToolbarButtonParameters()
                .text(ICON_FA_UPLOAD)
                .tooltip("Share your creature with other users:\nYour current creature will be uploaded to the server and made visible in the browser."))) {
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
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_SEEDLING).tooltip("Create a seed with current genome"))) {
        onCreateSeed();
    }

    AlienGui::Separator();
}

void CreatureEditorWindow::processTabWidget()
{
    if (ImGui::BeginTabBar("##", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown | ImGuiTabBarFlags_Reorderable)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            onScheduleAddTab(CreatureDescription());
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

void CreatureEditorWindow::onInjectGenome()
{
    auto const& tab = _tabs.at(_selectedTabIndex);
    auto creatureId = tab->getCreatureId();
    auto const& genome = tab->getGenome();
    auto success = _simulationFacade->changeCreature(genome);
    tab->onGenomeIntoCreatureInjected();
    if (success) {
        printOverlayMessage("Genome injected");
    } else {
        GenericMessageDialog::get().information("Error", "The genome could not be injected since the creature no longer exists.");
        tab->convertToDraftTab();
    }
}

void CreatureEditorWindow::onCreateSeed()
{
    auto pos = Viewport::get().getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;

    auto genome = _tabs.at(_selectedTabIndex)->getGenome();

    auto parameter = _simulationFacade->getSimulationParameters();
    auto numNodes = GenomeDescriptionInfoService::get().getNumberOfNodes(genome);
    auto energy = parameter.normalCellEnergy.value[EditorModel::get().getDefaultColorCode()] * toFloat(numNodes * 2 + 1);
    auto data =
        CollectionDescription()
            .creatures({
                genome,
            })
            .cells({
                CellDescription().pos(pos).energy(energy).stiffness(1.0f).color(EditorModel::get().getDefaultColorCode()).cellTypeData(ConstructorDescription()).creatureId(genome._id),
            });
    _simulationFacade->addAndSelectSimulationData(std::move(data));
    EditorModel::get().update();

    printOverlayMessage("Seed created");
}

void CreatureEditorWindow::onScheduleAddTab(CreatureDescription const& genome, std::optional<uint64_t> const& creatureId)
{
    if (creatureId.has_value()) {
        _tabToAdd = _CreatureTabWidget::createPinnedCreatureTab(_simulationFacade, genome, creatureId.value());
    } else {
        _tabToAdd = _CreatureTabWidget::createDraftCreatureTab(_simulationFacade, genome);
    }
}

void CreatureEditorWindow::pushStyleColorForTab(CreatureTabWidget const& creatureTab)
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
