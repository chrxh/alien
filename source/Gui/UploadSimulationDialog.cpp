#include "UploadSimulationDialog.h"

#include <imgui.h>

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "NetworkController.h"
#include "StyleRepository.h"
#include "BrowserWindow.h"

_UploadSimulationDialog::_UploadSimulationDialog(
    BrowserWindow const& browserWindow,
    SimulationController const& simController,
    NetworkController const& networkController)
    : _simController(simController)
    , _networkController(networkController)
    , _browserWindow(browserWindow)
{
    auto& settings = GlobalSettings::getInstance();
    _simName = settings.getStringState("dialogs.upload.simulation name", "");
    _simDescription = settings.getStringState("dialogs.upload.simulation description", "");
}

_UploadSimulationDialog::~_UploadSimulationDialog()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setStringState("dialogs.upload.simulation name", _simName);
    settings.setStringState("dialogs.upload.simulation description", _simDescription);
}

void _UploadSimulationDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Upload simulation");
    if (ImGui::BeginPopupModal("Upload simulation", NULL, ImGuiWindowFlags_None)) {
        AlienImGui::Text("Data privacy policy");
        AlienImGui::HelpMarker("The simulation file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");
        AlienImGui::Separator();

        AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Simulation name").textWidth(0), _simName);
        AlienImGui::Separator();
        AlienImGui::InputTextMultiline(
            AlienImGui::InputTextMultilineParameters()
                .hint("Description (optional)")
                .textWidth(0)
                .height(
                ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(50)),
            _simDescription);

        AlienImGui::Separator();

        ImGui::BeginDisabled(_simName.empty());
        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            onUpload();
        }
        ImGui::EndDisabled();
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            _simName = _origSimName;
            _simDescription = _origSimDescription;
        }

        ImGui::EndPopup();
    }
}

void _UploadSimulationDialog::show()
{
    _show = true;
    _origSimName = _simName;
    _origSimDescription = _simDescription;
}

void _UploadSimulationDialog::onUpload()
{
    DeserializedSimulation sim;
    sim.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
    sim.settings = _simController->getSettings();
    sim.symbolMap = _simController->getSymbolMap();
    sim.content = _simController->getClusteredSimulationData();

    std::string content, settings, symbolMap;
    if (!Serializer::serializeSimulationToStrings(content, settings, symbolMap, sim)) {
        MessageDialog::getInstance().show("Save simulation", "The simulation could not be uploaded.");
        return;
    }

    if (!_networkController->uploadSimulation(
        _simName,
        _simDescription,
        {sim.settings.generalSettings.worldSizeX, sim.settings.generalSettings.worldSizeY},
        sim.content.getNumberOfCellAndParticles(),
        content,
        settings,
        symbolMap)) {
        MessageDialog::getInstance().show("Error", "Failed to upload simulation.");
        return;
    }
    _browserWindow->onRefresh();
}
