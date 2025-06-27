#include "CreatureEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/CreatureDescriptionInfoService.h"

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
        if (!tab->isDraft() && tab->getCreatureDescription()._id == creature._id) {
            tabIndex = toInt(index);
        }
    }
    if (tabIndex) {
        _tabIndexToSelect = *tabIndex;
    } else {
        onScheduleAddTab(creature, false);
    }
}

CreatureEditorWindow::CreatureEditorWindow()
    : AlienWindow("Creature editor", "windows.creature editor", false, true, {500.0f, 300.0f})
{}

void CreatureEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    // Initialize the first tab with a draft creature
    _tabs.emplace_back(_CreatureTabWidget::createDraftCreatureTab(_simulationFacade, CreatureDescription()));
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
    if (ImGui::BeginTabBar("##CreatureTabWidget", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown | ImGuiTabBarFlags_Reorderable)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            onScheduleAddTab(CreatureDescription(), true);
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
    auto const& genome = tab->getCreatureDescription();
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

    auto creature = _tabs.at(_selectedTabIndex)->getCreatureDescription();

    auto parameter = _simulationFacade->getSimulationParameters();
    auto numNodes = CreatureDescriptionInfoService::get().getNumberOfNodes(creature);
    auto energy = parameter.normalCellEnergy.value[EditorModel::get().getDefaultColorCode()] * toFloat(numNodes * 2 + 1);
    auto data =
        CollectionDescription()
            .creatures({
                creature,
            })
            .cells({
                CellDescription().pos(pos).energy(energy).stiffness(1.0f).color(EditorModel::get().getDefaultColorCode()).cellTypeData(ConstructorDescription()).creatureId(creature._id),
            });
    _simulationFacade->addAndSelectSimulationData(std::move(data));
    EditorModel::get().update();

    printOverlayMessage("Seed created");
}

void CreatureEditorWindow::onScheduleAddTab(CreatureDescription const& creature, bool draft)
{
    if (draft) {
        _tabToAdd = _CreatureTabWidget::createDraftCreatureTab(_simulationFacade, creature);
    } else {
        _tabToAdd = _CreatureTabWidget::createSimulatedCreatureTab(_simulationFacade, creature);
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
