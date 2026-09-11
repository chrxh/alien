#include "ReplaceSimulationDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "GenomeEditorWindow.h"
#include "NetworkTransferController.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    std::string getResourceTypeString(NetworkResourceType resourceType)
    {
        return resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
    }
}

ReplaceSimulationDialog::ReplaceSimulationDialog()
    : AlienDialog("Replace", {660.0f, 500.0f})
{}

void ReplaceSimulationDialog::open(NetworkResourceType resourceType, BrowserLeaf const& leaf)
{
    changeTitle("Replace " + getResourceTypeString(resourceType));
    _resourceType = resourceType;
    _leaf = leaf;
    if (_resourceType == NetworkResourceType_Simulation) {
        _preview.createForSimulation();
    } else {
        _preview.createForGenome(GenomeEditorWindow::get().getCurrentPreviewDescs());
    }
    AlienDialog::open();
}

void ReplaceSimulationDialog::processIntern()
{
    if (ImGui::BeginChild("##content", {0, ImGui::GetContentRegionAvail().y - scale(50.0f)})) {
        auto resourceTypeString = getResourceTypeString(_resourceType);
        ImGui::TextWrapped(
            "Do you really want to replace the following %s with the currently opened %s?", resourceTypeString.c_str(), resourceTypeString.c_str());
        AlienGui::Text(AlienGui::TextParameters().text(_leaf.leafName).style(AlienGui::TextStyle::Bold));

        ImGui::Spacing();
        _preview.process();
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Yes")) {
        close();
        onReplace();
    }
    ImGui::SameLine();
    if (AlienGui::Button("No")) {
        close();
    }
}

void ReplaceSimulationDialog::onReplace()
{
    auto data = [&]() -> std::variant<ReplaceNetworkResourceRequestData::SimulationData, ReplaceNetworkResourceRequestData::CreatureData> {
        if (_resourceType == NetworkResourceType_Simulation) {
            return ReplaceNetworkResourceRequestData::SimulationData{
                .zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos(), .jpg = _preview.getJpg()};
        } else {
            return ReplaceNetworkResourceRequestData::CreatureData{.description = GenomeEditorWindow::get().getCurrentGenome(), .jpg = _preview.getJpg()};
        }
    }();
    NetworkTransferController::get().onReplace(ReplaceNetworkResourceRequestData{
        .resourceId = _leaf.rawTO->id, .workspaceType = _leaf.rawTO->workspaceType, .downloadCache = BrowserWindow::get().getSimulationCache(), .data = data});
}
