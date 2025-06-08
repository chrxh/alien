#include "CreatureEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "AlienImGui.h"
#include "CreatureTabLayoutData.h"
#include "CreatureTabWidget.h"
#include "EditorController.h"

CreatureEditorWindow::CreatureEditorWindow()
    : AlienWindow("Creature editor", "windows.creature editor", false, true, {500.0f, 300.0f})
{
}

void CreatureEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    //_creatureTabLayoutData = std::make_shared<CreatureTabLayoutData>();

    //_creatureTabLayoutData->_genomeEditorWidth = GlobalSettings::get().getValue("windows.creature editor.genome editor width", scale(GenomeEditorWidth));
    //_creatureTabLayoutData->_geneEditorWidth = GlobalSettings::get().getValue("windows.creature editor.gene editor width", scale(GeneEditorWidth));
    //_creatureTabLayoutData->_previewsHeight = GlobalSettings::get().getValue("windows.creature editor.previews height", scale(PreviewsHeight));
    //_creatureTabLayoutData->_desiredConfigurationPreviewWidth =
    //    GlobalSettings::get().getValue("windows.creature editor.desired configuration preview width", scale(DesiredConfigurationPreviewWidth));

    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    scheduleAddTab(genome);
}

void CreatureEditorWindow::shutdownIntern()
{
    //GlobalSettings::get().setValue("windows.creature editor.genome editor width", _creatureTabLayoutData->_genomeEditorWidth);
    //GlobalSettings::get().setValue("windows.creature editor.gene editor width", _creatureTabLayoutData->_geneEditorWidth);
    //GlobalSettings::get().setValue("windows.creature editor.previews height", _creatureTabLayoutData->_previewsHeight);
    //GlobalSettings::get().setValue("windows.creature editor.desired configuration preview width", _creatureTabLayoutData->_desiredConfigurationPreviewWidth);
}

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
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN))) {
    }
    AlienImGui::Tooltip("Open creature from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SAVE))) {
    }
    AlienImGui::Tooltip("Save creature to file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_UPLOAD))) {
    }
    AlienImGui::Tooltip("Share your creature with other users:\nYour current creature will be uploaded to the server and made visible in the browser.");

    AlienImGui::Separator();
}

void CreatureEditorWindow::processTabWidget()
{
    if (ImGui::BeginTabBar("##", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            scheduleAddTab(GenomeDescription_New());
        }
        AlienImGui::Tooltip("New creature");

        std::optional<int> tabToDelete;

        // Process tabs
        for (auto const& [index, creatureTab] : _tabs | boost::adaptors::indexed(0)) {

            bool open = true;
            bool* openPtr = nullptr;
            if (_tabs.size() > 1) {
                openPtr = &open;
            }
            int flags = ImGuiTabItemFlags_None;
            if (ImGui::BeginTabItem(creatureTab->getName().c_str(), openPtr, flags)) {
                _selectedTabIndex = toInt(index);
                creatureTab->process();
                ImGui::EndTabItem();
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

        // Add tab
        if (_tabToAdd.has_value()) {
            _tabs.emplace_back(_tabToAdd.value());
            _tabToAdd.reset();
        }

        ImGui::EndTabBar();
    }
}

void CreatureEditorWindow::scheduleAddTab(GenomeDescription_New const& genome)
{
    _tabToAdd = _CreatureTabWidget::createDraftCreatureTab(genome);
}
