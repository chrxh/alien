#include "UploadSimulationDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "NetworkController.h"
#include "StyleRepository.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"
#include "Viewport.h"

_UploadSimulationDialog::_UploadSimulationDialog(
    BrowserWindow const& browserWindow,
    SimulationController const& simController,
    NetworkController const& networkController,
    Viewport const& viewport)
    : _AlienDialog("Upload simulation")
    ,_simController(simController)
    , _networkController(networkController)
    , _browserWindow(browserWindow)
    , _viewport(viewport)
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

void _UploadSimulationDialog::processIntern()
{
    AlienImGui::Text("Data privacy policy");
    AlienImGui::HelpMarker("The simulation file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");
    AlienImGui::Separator();

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint("Simulation name").textWidth(0), _simName);
    AlienImGui::Separator();
    AlienImGui::InputTextMultiline(
        AlienImGui::InputTextMultilineParameters()
            .hint("Description (optional)")
            .textWidth(0)
            .height(ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scale(50)),
        _simDescription);

    AlienImGui::Separator();

    ImGui::BeginDisabled(_simName.empty());
    if (AlienImGui::Button("OK")) {
        close();
        onUpload();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
        _simName = _origSimName;
        _simDescription = _origSimDescription;
    }
}

void _UploadSimulationDialog::openIntern()
{
    _origSimName = _simName;
    _origSimDescription = _simDescription;
}

void _UploadSimulationDialog::onUpload()
{
    printOverlayMessage("Uploading ...");

    delayedExecution([=, this] {
        DeserializedSimulation deserializedSim;
        deserializedSim.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        deserializedSim.auxiliaryData.zoom = _viewport->getZoomFactor();
        deserializedSim.auxiliaryData.center = _viewport->getCenterInWorldPos();
        deserializedSim.auxiliaryData.generalSettings = _simController->getGeneralSettings();
        deserializedSim.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
        deserializedSim.mainData = _simController->getClusteredSimulationData();

        SerializedSimulation serializedSim;
        if (!Serializer::serializeSimulationToStrings(serializedSim, deserializedSim)) {
            MessageDialog::getInstance().show("Save simulation", "The simulation could not be uploaded.");
            return;
        }

        if (!_networkController->uploadSimulation(
                _simName,
                _simDescription,
                {deserializedSim.auxiliaryData.generalSettings.worldSizeX, deserializedSim.auxiliaryData.generalSettings.worldSizeY},
                deserializedSim.mainData.getNumberOfCellAndParticles(),
                serializedSim.mainData,
                serializedSim.auxiliaryData)) {
            MessageDialog::getInstance().show("Error", "Failed to upload simulation.");
            return;
        }
        _browserWindow->onRefresh();
    });
}
