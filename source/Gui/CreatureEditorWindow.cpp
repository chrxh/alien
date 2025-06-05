#include "CreatureEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "AlienImGui.h"
#include "CreatureTabLayoutData.h"
#include "CreatureTabWidget.h"
#include "EditorController.h"

namespace
{
    auto constexpr GenomeEditorWidth = 300.0f;
    auto constexpr GeneEditorWidth = 300.0f;
    auto constexpr PreviewsHeight = 200.0f;
    auto constexpr DesiredConfigurationPreviewWidth = 300.0f;
}

CreatureEditorWindow::CreatureEditorWindow()
    : AlienWindow("Creature editor", "windows.creature editor", false, true)
{
}

void CreatureEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    _creatureTabLayoutData = std::make_shared<_CreatureTabLayoutData>();

    _creatureTabLayoutData->_genomeEditorWidth = GlobalSettings::get().getValue("windows.creature editor.genome editor width", scale(GenomeEditorWidth));
    _creatureTabLayoutData->_geneEditorWidth = GlobalSettings::get().getValue("windows.creature editor.gene editor width", scale(GeneEditorWidth));
    _creatureTabLayoutData->_previewsHeight = GlobalSettings::get().getValue("windows.creature editor.previews height", scale(PreviewsHeight));
    _creatureTabLayoutData->_desiredConfigurationPreviewWidth =
        GlobalSettings::get().getValue("windows.creature editor.desired configuration preview width", scale(DesiredConfigurationPreviewWidth));

    scheduleAddTab(GenomeDescription_New());
}

void CreatureEditorWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.creature editor.genome editor width", _creatureTabLayoutData->_genomeEditorWidth);
    GlobalSettings::get().setValue("windows.creature editor.gene editor width", _creatureTabLayoutData->_geneEditorWidth);
    GlobalSettings::get().setValue("windows.creature editor.previews height", _creatureTabLayoutData->_previewsHeight);
    GlobalSettings::get().setValue("windows.creature editor.desired configuration preview width", _creatureTabLayoutData->_desiredConfigurationPreviewWidth);
}

void CreatureEditorWindow::processIntern()
{
    correctingLayout();

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
        for (auto const& [index, tab] : _tabs | boost::adaptors::indexed(0)) {

            bool open = true;
            bool* openPtr = nullptr;
            if (_tabs.size() > 1) {
                openPtr = &open;
            }
            int flags = ImGuiTabItemFlags_None;
            if (ImGui::BeginTabItem(("Creature " + std::to_string(index + 1)).c_str(), openPtr, flags)) {
                _selectedTabIndex = toInt(index);
                tab->process();
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
    _tabToAdd = std::make_shared<_CreatureTabWidget>(genome, _creatureTabLayoutData);
}

void CreatureEditorWindow::correctingLayout()
{
    if (_lastGenomeEditorWidth.has_value()) {
        _creatureTabLayoutData->_geneEditorWidth += _lastGenomeEditorWidth.value() - _creatureTabLayoutData->_genomeEditorWidth;
    }

    auto windowSize = ImGui::GetWindowSize();
    if (_lastWindowSize.has_value() && _lastWindowSize->x > 0 && _lastWindowSize->y > 0) {
        if (_lastWindowSize->x != windowSize.x || _lastWindowSize->y != windowSize.y) {
            auto scalingX = windowSize.x / _lastWindowSize->x;
            auto scalingY = windowSize.y / _lastWindowSize->y;
            _creatureTabLayoutData->_genomeEditorWidth *= scalingX;
            _creatureTabLayoutData->_geneEditorWidth *= scalingX;
            _creatureTabLayoutData->_desiredConfigurationPreviewWidth *= scalingX;
            _creatureTabLayoutData->_previewsHeight *= scalingY;
        }
    }
    _lastWindowSize = {windowSize.x, windowSize.y};

    _creatureTabLayoutData->_genomeEditorWidth = std::max(scale(50.0f), _creatureTabLayoutData->_genomeEditorWidth);
    _creatureTabLayoutData->_geneEditorWidth = std::max(scale(50.0f), _creatureTabLayoutData->_geneEditorWidth);
    _creatureTabLayoutData->_desiredConfigurationPreviewWidth = std::max(scale(50.0f), _creatureTabLayoutData->_desiredConfigurationPreviewWidth);
    _creatureTabLayoutData->_previewsHeight =
        std::min(ImGui::GetContentRegionAvail().y - scale(50.0f), std::max(scale(50.0f), _creatureTabLayoutData->_previewsHeight));

    _lastGenomeEditorWidth = _creatureTabLayoutData->_genomeEditorWidth;
}
