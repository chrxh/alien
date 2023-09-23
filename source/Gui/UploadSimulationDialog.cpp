#include "UploadSimulationDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "NetworkController.h"
#include "StyleRepository.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"
#include "Viewport.h"
#include "GenomeEditorWindow.h"
#include "LoginDialog.h"

namespace
{
    std::map<DataType, std::string> const BrowserDataTypeToLowerString = {
        {DataType_Simulation, "simulation"},
        {DataType_Genome, "genome"}};
    std::map<DataType, std::string> const BrowserDataTypeToUpperString = {
        {DataType_Simulation, "Simulation"},
        {DataType_Genome, "Genome"}};
}

_UploadSimulationDialog::_UploadSimulationDialog(
    BrowserWindow const& browserWindow,
    LoginDialog const& loginDialog,
    SimulationController const& simController,
    NetworkController const& networkController,
    Viewport const& viewport,
    GenomeEditorWindow const& genomeEditorWindow)
    : _AlienDialog("")
    , _simController(simController)
    , _networkController(networkController)
    , _browserWindow(browserWindow)
    , _loginDialog(loginDialog)
    , _viewport(viewport)
    , _genomeEditorWindow(genomeEditorWindow)
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

void _UploadSimulationDialog::open(DataType dataType)
{
    if (_networkController->getLoggedInUserName()) {
        changeTitle("Upload " + BrowserDataTypeToLowerString.at(dataType));
        _dataType = dataType;
        _AlienDialog::open();
    } else {
        _loginDialog->open();
    }
}

void _UploadSimulationDialog::processIntern()
{
    AlienImGui::Text("Data privacy policy");
    AlienImGui::HelpMarker(
        "The " + BrowserDataTypeToLowerString.at(_dataType)
        + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");
    AlienImGui::Separator();

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_dataType)  + " name").textWidth(0), _simName);
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
        std::string mainData;
        std::string auxiliaryData;
        IntVector2D size;
        int numObjects = 0;

        if (_dataType == DataType_Simulation) {
            DeserializedSimulation deserializedSim;
            deserializedSim.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
            deserializedSim.auxiliaryData.zoom = _viewport->getZoomFactor();
            deserializedSim.auxiliaryData.center = _viewport->getCenterInWorldPos();
            deserializedSim.auxiliaryData.generalSettings = _simController->getGeneralSettings();
            deserializedSim.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
            deserializedSim.mainData = _simController->getClusteredSimulationData();

            SerializedSimulation serializedSim;
            if (!Serializer::serializeSimulationToStrings(serializedSim, deserializedSim)) {
                MessageDialog::getInstance().information(
                    "Upload simulation", "The simulation could not be serialized for uploading.");
                return;
            }
            mainData = serializedSim.mainData;
            auxiliaryData = serializedSim.auxiliaryData;
            size = {deserializedSim.auxiliaryData.generalSettings.worldSizeX, deserializedSim.auxiliaryData.generalSettings.worldSizeY};
            numObjects = deserializedSim.mainData.getNumberOfCellAndParticles();
        } else {
            auto genome = _genomeEditorWindow->getCurrentGenome();
            if (genome.cells.empty()) {
                showMessage("Upload genome", "The is no valid genome in the genome editor selected.");
                return;
            }
            auto genomeData = GenomeDescriptionConverter::convertDescriptionToBytes(genome);
            numObjects = GenomeDescriptionConverter::getNumNodesRecursively(genomeData);

            if (!Serializer::serializeGenomeToString(mainData, genomeData)) {
                showMessage("Upload genome", "The genome could not be serialized for uploading.");
                return;
            }
        }

        if (!_networkController->uploadSimulation(_simName, _simDescription, size, numObjects, mainData, auxiliaryData, _dataType)) {
            showMessage("Error", "Failed to upload " + BrowserDataTypeToLowerString.at(_dataType) + ".");
            return;
        }
        _browserWindow->onRefresh();
    });
}
