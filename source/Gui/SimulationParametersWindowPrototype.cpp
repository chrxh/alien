#include "SimulationParametersWindowPrototype.h"

#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/SimulationFacade.h"

#include "AlienImGui.h"
#include "OverlayController.h"

namespace
{
    auto constexpr MasterEditorHeight = 200.0f;
    auto constexpr DetailEditorHeight = 200.0f;
    auto constexpr AddonHeight = 200.0f;

}

SimulationParametersWindowPrototype::SimulationParametersWindowPrototype()
    : AlienWindow("Simulation parameters Prototype", "windows.simulation parameters prototype", false)
{}

void SimulationParametersWindowPrototype::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    _masterHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.master height", scale(MasterEditorHeight));
    //_detailHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.detail height", scale(DetailEditorHeight));
    _addonHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.addon height", scale(AddonHeight));
}

void SimulationParametersWindowPrototype::processIntern()
{
    processToolbar();

    auto masterEditorHeight = [&] {
        if (_masterOpen) {
            return _masterHeight;
        } else {
            return scale(25.0f);
        }
    }();
    if (ImGui::BeginChild("##masterEditor", {0, masterEditorHeight})) {
        //processMasterEditor();
        ImGui::Button("Test3", {ImGui::GetContentRegionAvail().x, masterEditorHeight});
    }
    ImGui::EndChild();
    if (_masterOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushID("master");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters(), _masterHeight);
        ImGui::PopID();
    }

    auto detailEditorHeight = [&] {
        if (_detailOpen) {
            return /*std::max(scale(25.0f), */ImGui::GetContentRegionAvail().y - _addonHeight;
        } else {
            return scale(25.0f);
        }
    }();
    if (ImGui::BeginChild("##detailEditor", {0, detailEditorHeight})) {
        //processDetailEditor();
        ImGui::Button("Test2", {ImGui::GetContentRegionAvail().x, detailEditorHeight});
    }
    ImGui::EndChild();
    if (_detailOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushID("detail");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters().additive(false), _addonHeight);
        ImGui::PopID();
    }

    if (ImGui::BeginChild("##addon", {0, -scale(50.0f)})) {
        //processAddonList();
        ImGui::Button("Test", {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y});
    }
    ImGui::EndChild();

    processStatusBar();

    validateAndCorrect();
}

void SimulationParametersWindowPrototype::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.simulation parameters prototype.master height", _masterHeight);
    //GlobalSettings::get().setValue("windows.simulation parameters prototype.detail height", _detailHeight);
    GlobalSettings::get().setValue("windows.simulation parameters prototype.addon height", _addonHeight);
}

void SimulationParametersWindowPrototype::processToolbar()
{
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
    }
    AlienImGui::Tooltip("Open simulation parameters from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
    }
    AlienImGui::Tooltip("Save simulation parameters to file");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedParameters = _simulationFacade->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }
    AlienImGui::Tooltip("Copy simulation parameters");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        _simulationFacade->setSimulationParameters(*_copiedParameters);
        _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste simulation parameters");

    AlienImGui::Separator();
}

void SimulationParametersWindowPrototype::processMasterEditor()
{
    if(_masterOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Region").highlighted(true).defaultOpen(_masterOpen))) {
        //if (ImGui::BeginChild("##masterChildWindow", {0, scale(_masterHeight)})) {
        //}        
        //ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void SimulationParametersWindowPrototype::processDetailEditor()
{
    if (_detailOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Parameters").highlighted(true).defaultOpen(_detailOpen))) {
        //if (ImGui::BeginChild("##detailChildWindow", {0, scale(_detailHeight)})) {
        //}
        //ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void SimulationParametersWindowPrototype::processAddonList()
{
    if (_addonOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addons").highlighted(true).defaultOpen(_addonOpen))) {
        //if (ImGui::BeginChild("##detailChildWindow", {0, scale(_detailHeight)})) {
        //}
        //ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void SimulationParametersWindowPrototype::processStatusBar()
{
    std::vector<std::string> statusItems;
    statusItems.emplace_back("CTRL + click on a slider to type in a precise value");

    AlienImGui::StatusBar(statusItems);
}

void SimulationParametersWindowPrototype::validateAndCorrect()
{
    //auto windowHeight = ImGui::GetWindowSize().y;
    //auto sumHeights = 50.0f + _masterHeight + _detailHeight + _addonHeight + 50.0f + 90.0f;
    //if (sumHeights > windowHeight) {
    //    printf("sumHeights: %f, windowHeight: %f\n", sumHeights, windowHeight);
    //    auto diff = sumHeights - windowHeight;
    //    auto factor = 1.0f - diff / (_masterHeight + _detailHeight + _addonHeight);
    //    _masterHeight *= factor;
    //    _detailHeight *= factor;
    //    _addonHeight *= factor;
    //} else {
    //    //_addonHeight += windowHeight - sumHeights;
    //}
}
