#include "UploadSimulationDialog.h"

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "Network/NetworkService.h"
#include "Network/ValidationService.h"

#include "AlienImGui.h"
#include "MessageDialog.h"
#include "StyleRepository.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"
#include "Viewport.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "SerializationHelperService.h"

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
    GenomeEditorWindow const& genomeEditorWindow)
    : _AlienDialog("")
    , _simController(simController)
    , _browserWindow(browserWindow)
    , _loginDialog(loginDialog)
    , _genomeEditorWindow(genomeEditorWindow)
{
    auto& settings = GlobalSettings::getInstance();
    _share = settings.getBool("dialogs.upload.share", _share);
}

_UploadSimulationDialog::~_UploadSimulationDialog()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setBool("dialogs.upload.share", _share);
}

void _UploadSimulationDialog::open(NetworkResourceType resourceType, std::string const& folder)
{
    if (NetworkService::getLoggedInUserName()) {
        changeTitle("Upload " + BrowserDataTypeToLowerString.at(resourceType));
        _resourceType = resourceType;
        _folder = folder;
        _resourceName = _resourceNameByFolder[_folder];
        _resourceDescription = _resourceDescriptionByFolder[_folder];
        _AlienDialog::open();
    } else {
        _loginDialog->open();
    }
}

void _UploadSimulationDialog::processIntern()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);
    if (ImGui::BeginChild("##header", ImVec2(0, scale(52.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
        AlienImGui::Text("Data privacy policy");
        AlienImGui::HelpMarker(
            "The " + resourceTypeString + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");

        AlienImGui::Text("How to use or create folders?");
        AlienImGui::HelpMarker(
            "If you want to upload the " + resourceTypeString
            + " to a folder, you can use the `/`-notation. The folder will be created automatically if it does not exist.\nFor instance, naming a simulation "
              "as `Biome/Water "
              "world/Initial/Variant 1` will create the nested folders `Biome`, `Water world` and `Initial`.");
    }
    ImGui::EndChild();

    if (!_folder.empty()) {
        if (ImGui::BeginChild("##folder info", ImVec2(0, scale(85.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
            AlienImGui::Text("The following folder has been selected in the browser\nand will used for the upload:\n\n");
            AlienImGui::BoldText(_folder);
        }
        ImGui::EndChild();
    }

    AlienImGui::Separator();

    AlienImGui::InputText(AlienImGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType)  + " name").textWidth(0), _resourceName);

    AlienImGui::Separator();

    ImGui::PushID("description");
    AlienImGui::InputTextMultiline(
        AlienImGui::InputTextMultilineParameters()
            .hint("Description (optional)")
            .textWidth(0)
            .height(ImGui::GetContentRegionAvail().y - StyleRepository::get().scale(70.0f)),
        _resourceDescription);
    ImGui::PopID();

    AlienImGui::ToggleButton(
        AlienImGui::ToggleButtonParameters()
            .name("Make public")
            .tooltip(
                "If true, the " + resourceTypeString + " will be visible to all users. If false, the " + resourceTypeString
                + " will only be visible in the private workspace. This property can also be changed later if desired."),
        _share);

    AlienImGui::Separator();

    ImGui::BeginDisabled(_resourceName.empty());
    if (AlienImGui::Button("OK")) {
        if (ValidationService::isStringValidForDatabase(_resourceName) && ValidationService::isStringValidForDatabase(_resourceDescription)) {
            close();
            onUpload();
        } else {
            showMessage("Error", Const::NotAllowedCharacters);
        }
        _resourceNameByFolder[_folder] = _resourceName;
        _resourceDescriptionByFolder[_folder] = _resourceDescription;
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
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

        DeserializedSimulation deserializedSim;
        if (_resourceType == NetworkResourceType_Simulation) {
            deserializedSim = SerializationHelperService::getDeserializedSerialization(_simController);

            SerializedSimulation serializedSim;
            if (!SerializerService::serializeSimulationToStrings(serializedSim, deserializedSim)) {
                MessageDialog::get().information(
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

        std::string resourceId;
        auto workspaceType = _share ? WorkspaceType_Public : WorkspaceType_Private;
        if (!NetworkService::uploadResource(
                resourceId, _folder + _resourceName, _resourceDescription, size, numObjects, mainData, settings, statistics, _resourceType, workspaceType)) {
            showMessage(
                "Error",
                "Failed to upload " + BrowserDataTypeToLowerString.at(_resourceType)
                    + ".\n\nPossible reasons:\n\n" ICON_FA_CHEVRON_RIGHT " The server is not reachable.\n\n" ICON_FA_CHEVRON_RIGHT " The total size of your uploads exceeds the allowed storage limit.");
            return;
        }
        if (_resourceType == NetworkResourceType_Simulation) {
            _browserWindow->getSimulationCache().insertOrAssign(resourceId, deserializedSim);
        }
        _browserWindow->onRefresh();
    });
}
