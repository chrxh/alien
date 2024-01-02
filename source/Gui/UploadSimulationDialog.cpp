#include "UploadSimulationDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "StyleRepository.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"
#include "Viewport.h"
#include "GenomeEditorWindow.h"
#include "LoginDialog.h"

namespace
{
    auto constexpr FolderWidgetHeight = 50.0f;

    std::map<NetworkResourceType, std::string> const BrowserDataTypeToLowerString = {
        {NetworkResourceType_Simulation, "simulation"},
        {NetworkResourceType_Genome, "genome"}};
    std::map<NetworkResourceType, std::string> const BrowserDataTypeToUpperString = {
        {NetworkResourceType_Simulation, "Simulation"},
        {NetworkResourceType_Genome, "Genome"}};
}

_UploadSimulationDialog::_UploadSimulationDialog(
    BrowserWindow const& browserWindow,
    LoginDialog const& loginDialog,
    SimulationController const& simController,
    Viewport const& viewport,
    GenomeEditorWindow const& genomeEditorWindow)
    : _AlienDialog("")
    , _simController(simController)
    , _browserWindow(browserWindow)
    , _loginDialog(loginDialog)
    , _viewport(viewport)
    , _genomeEditorWindow(genomeEditorWindow)
{
    auto& settings = GlobalSettings::getInstance();
    _resourceName = settings.getStringState("dialogs.upload.simulation name", "");
    _resourceDescription = settings.getStringState("dialogs.upload.simulation description", "");
}

_UploadSimulationDialog::~_UploadSimulationDialog()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setStringState("dialogs.upload.simulation name", _resourceName);
    settings.setStringState("dialogs.upload.simulation description", _resourceDescription);
}

void _UploadSimulationDialog::open(NetworkResourceType resourceType, std::string const& folder)
{
    auto& networkService = NetworkService::getInstance();
    if (networkService.getLoggedInUserName()) {
        auto workspaceType = _browserWindow->getCurrentWorkspaceType();
        if (workspaceType == WorkspaceType_AlienProject && *networkService.getLoggedInUserName() != "alien-project") {
            MessageDialog::getInstance().information(
                "Upload " + BrowserDataTypeToLowerString.at(resourceType),
                "You are not allowed to upload to alien-project's workspace.\nPlease choose the public or private workspace in the browser.");
            return;
        }

        changeTitle("Upload " + BrowserDataTypeToLowerString.at(resourceType));
        _resourceType = resourceType;
        _workspaceType = workspaceType;
        _folder = folder;
        _AlienDialog::open();
    } else {
        _loginDialog->open();
    }
}

void _UploadSimulationDialog::processIntern()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);
    AlienImGui::Text("Data privacy policy");
    AlienImGui::HelpMarker(
        "The " + resourceTypeString + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");

    AlienImGui::Text("How to use or create folders?");
    AlienImGui::HelpMarker("If you want to upload the " + resourceTypeString
        + " to a folder, you can use the `/`-notation. The folder will be created automatically if it does not exist.\nFor instance, naming a simulation as `Biome/Water "
          "world/Initial/Variant 1` will create the nested folders `Biome`, `Water world` and `Initial`.");

    AlienImGui::Separator();

    if (!_folder.empty()) {
        std::string text = "The following folder has been selected and will used for the upload:\n" + _folder;
        ImGui::PushID("folder info");
        ImGui::BeginDisabled();
        AlienImGui::InputTextMultiline(AlienImGui::InputTextMultilineParameters().hint(_folder).textWidth(0).height(FolderWidgetHeight), text);
        ImGui::EndDisabled();
        ImGui::PopID();
    }

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType)  + " name").textWidth(0), _resourceName);

    AlienImGui::Separator();

    ImGui::PushID("description");
    AlienImGui::InputTextMultiline(
        AlienImGui::InputTextMultilineParameters()
            .hint("Description (optional)")
            .textWidth(0)
            .height(ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scale(50.0f)),
        _resourceDescription);
    ImGui::PopID();

    AlienImGui::Separator();

    ImGui::BeginDisabled(_resourceName.empty());
    if (AlienImGui::Button("OK")) {
        close();
        onUpload();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
        _resourceName = _origResourceName;
        _resourceDescription = _origResourceDescription;
    }
}

void _UploadSimulationDialog::openIntern()
{
    _origResourceName = _resourceName;
    _origResourceDescription = _resourceDescription;
}

void _UploadSimulationDialog::onUpload()
{
    printOverlayMessage("Uploading ...");

    delayedExecution([=, this] {
        std::string mainData;
        std::string settings;
        std::string statistics;
        IntVector2D size;
        int numObjects = 0;

        if (_resourceType == NetworkResourceType_Simulation) {
            DeserializedSimulation deserializedSim;
            deserializedSim.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
            deserializedSim.auxiliaryData.zoom = _viewport->getZoomFactor();
            deserializedSim.auxiliaryData.center = _viewport->getCenterInWorldPos();
            deserializedSim.auxiliaryData.generalSettings = _simController->getGeneralSettings();
            deserializedSim.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
            deserializedSim.statistics = _simController->getStatisticsHistory().getCopiedData();
            deserializedSim.mainData = _simController->getClusteredSimulationData();

            SerializedSimulation serializedSim;
            if (!SerializerService::serializeSimulationToStrings(serializedSim, deserializedSim)) {
                MessageDialog::getInstance().information(
                    "Upload simulation", "The simulation could not be serialized for uploading.");
                return;
            }
            mainData = serializedSim.mainData;
            settings = serializedSim.auxiliaryData;
            statistics = serializedSim.statistics;
            size = {deserializedSim.auxiliaryData.generalSettings.worldSizeX, deserializedSim.auxiliaryData.generalSettings.worldSizeY};
            numObjects = deserializedSim.mainData.getNumberOfCellAndParticles();
        } else {
            auto genome = _genomeEditorWindow->getCurrentGenome();
            if (genome.cells.empty()) {
                showMessage("Upload genome", "The is no valid genome in the genome editor selected.");
                return;
            }
            auto genomeData = GenomeDescriptionService::convertDescriptionToBytes(genome);
            numObjects = GenomeDescriptionService::getNumNodesRecursively(genomeData, true);

            if (!SerializerService::serializeGenomeToString(mainData, genomeData)) {
                showMessage("Upload genome", "The genome could not be serialized for uploading.");
                return;
            }
        }

        auto& networkService = NetworkService::getInstance();
        if (!networkService.uploadSimulation(_folder + _resourceName, _resourceDescription, size, numObjects, mainData, settings, statistics, _resourceType, _workspaceType)) {
            showMessage("Error", "Failed to upload " + BrowserDataTypeToLowerString.at(_resourceType) + ".");
            return;
        }
        _browserWindow->onRefresh();
    });
}
