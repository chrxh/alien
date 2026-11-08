#include "SaveSimulationDialog.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <vector>

#include <boost/algorithm/string/trim.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>
#include <Network/NetworkValidationService.h>

#include <Persister/Interface/SerializerService.h>
#include <Persister/Interface/SimulationLibraryService.h>

#include "AlienGui.h"
#include "BrowserController.h"
#include "BrowserData.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "FileTransferController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "NewSimulationService.h"
#include "SimulationLibraryController.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr PreviewWidth = 450.0f;
    auto constexpr ColumnSpacing = 24.0f;
    auto constexpr FooterHeight = 50.0f;
    auto constexpr MinColumnHeight = 100.0f;
    auto constexpr MinDescriptionHeight = 50.0f;
    auto constexpr SectionSpacing = 10.0f;
    auto constexpr TargetSectionHeight = 164.0f;
    auto constexpr ReplaceSectionHeight = 90.0f;
    auto constexpr MinMessageWidth = 60.0f;
    auto constexpr TimestampLengthWithoutSeconds = 16;  // e.g. 2026-11-13 14:02
    auto constexpr DateLength = 10;

    std::map<NetworkResourceType, std::string> const BrowserDataTypeToLowerString = {
        {NetworkResourceType_Simulation, "simulation"},
        {NetworkResourceType_Genome, "genome"}};
    std::map<NetworkResourceType, std::string> const BrowserDataTypeToUpperString = {
        {NetworkResourceType_Simulation, "Simulation"},
        {NetworkResourceType_Genome, "Genome"}};

    std::string getSuggestedNameWithoutFolder(NetworkResourceType resourceType)
    {
        if (resourceType == NetworkResourceType_Genome) {
            return GenomeEditorWindow::get().getCurrentGenome()._name;
        }
        auto projectName = std::string(_SimulationFacade::get()->getSimulationParameters().projectName.value);
        if (projectName == SimulationParameters().projectName.value) {
            return {};
        }
        return projectName;
    }

    // A simulation of this computer keeps its name. Uploads are collected in a folder with the name of the user.
    std::string getSuggestedName(NetworkResourceType resourceType, SaveTarget target)
    {
        auto name = getSuggestedNameWithoutFolder(resourceType);
        if (target == SaveTarget::ThisComputer) {
            return resourceType == NetworkResourceType_Simulation ? NewSimulationService::get().getNameInLibrary().value_or(name) : name;
        }
        auto userName = NetworkService::get().getLoggedInUserName();
        return userName.has_value() ? *userName + "/" + name : name;
    }

    std::string getFolder(std::string const& name)
    {
        auto& resourceService = NetworkResourceService::get();
        return resourceService.concatenateFolderName(resourceService.getFolderNames(name), true);
    }

    bool isServerTarget(SaveTarget target)
    {
        return target == SaveTarget::Community || target == SaveTarget::Private;
    }

    WorkspaceType toWorkspaceType(SaveTarget target)
    {
        return target == SaveTarget::Private ? WorkspaceType_Private : WorkspaceType_Public;
    }

    // The uploads that only the user sees are a place to save to, and one of two visibilities of an upload
    std::string getTargetLabel(SaveTarget target, NetworkResourceType resourceType, bool asPlace)
    {
        switch (target) {
        case SaveTarget::ThisComputer:
            return resourceType == NetworkResourceType_Simulation ? ICON_FA_DESKTOP "  My simulations" : ICON_FA_DESKTOP "  My genomes";
        case SaveTarget::Community:
            return ICON_FA_GLOBE "  Community";
        case SaveTarget::OtherLocation:
            return ICON_FA_FOLDER_OPEN "  Other location";
        default:
            return asPlace ? ICON_FA_LOCK "  My uploads" : ICON_FA_LOCK "  Only me";
        }
    }

    std::string getTargetExplanation(SaveTarget target, std::string const& resourceTypeString)
    {
        switch (target) {
        case SaveTarget::ThisComputer:
            return "The " + resourceTypeString + " is stored on this computer. You can share it from the browser at any time.";
        case SaveTarget::Community:
            return "Everyone can see and download the " + resourceTypeString
                + " in the Community workspace. It is also announced on the ALIEN Discord. Work in progress is welcome.";
        case SaveTarget::OtherLocation:
            return "The " + resourceTypeString + " is stored as a file in any folder. Save asks for the folder and the file name.";
        default:
            return "The " + resourceTypeString
                + " is stored on the server, where only you can see it. It appears under My uploads in the browser and can be shared later.";
        }
    }

    SimulationFile const* findLibraryFile(NetworkResourceType resourceType, std::string const& name)
    {
        auto const& content = SimulationLibraryController::get().getContent();
        auto const& files = resourceType == NetworkResourceType_Simulation ? content.simulations : content.genomes;
        auto findResult = std::ranges::find_if(files, [&](SimulationFile const& file) { return file.name == name; });
        return findResult != files.end() ? &*findResult : nullptr;
    }

    // The private workspace lists all uploads of the user, whoever can see them
    NetworkResourceRawTO findOwnUpload(NetworkResourceType resourceType, SaveTarget target, std::string const& name)
    {
        auto userName = NetworkService::get().getLoggedInUserName();
        if (!userName.has_value()) {
            return nullptr;
        }
        auto const& rawTOs = BrowserController::get().getData()->getWorkspace(WorkspaceId{resourceType, WorkspaceType_Private}).rawTOs;
        auto findResult = std::ranges::find_if(rawTOs, [&](NetworkResourceRawTO const& rawTO) {
            return rawTO->userName == *userName && rawTO->resourceName == name && rawTO->workspaceType == toWorkspaceType(target);
        });
        return findResult != rawTOs.end() ? *findResult : nullptr;
    }

    void processLabel(std::string const& text, std::optional<std::string> const& annotation = std::nullopt)
    {
        auto labelAscent = ImGui::GetFont()->Ascent;
        AlienGui::Text(text);
        if (annotation.has_value()) {
            ImGui::SameLine();
            ImGui::PushFont(StyleService::get().getTinyFont());
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + labelAscent - ImGui::GetFont()->Ascent);
            AlienGui::Text(AlienGui::TextParameters().text(*annotation).style(AlienGui::TextStyle::Decent));
            ImGui::PopFont();
        }
    }

    void addSectionSpacing()
    {
        ImGui::Dummy({0, scale(SectionSpacing)});
    }
}

void SaveSimulationDialog::open(NetworkResourceType resourceType, SaveTarget target, std::string const& suggestedName)
{
    if (isServerTarget(target) && !NetworkService::get().getLoggedInUserName()) {
        LoginDialog::get().open();
        return;
    }
    _mode = Mode::RunningResource;
    _resourceType = resourceType;
    _targets = {SaveTarget::ThisComputer, SaveTarget::Community, SaveTarget::Private, SaveTarget::OtherLocation};
    _target = target;
    _simulationFile.reset();
    _savedGenome.reset();
    _replacedUpload.reset();
    _errorMessage.reset();

    if (suggestedName.empty()) {
        _name = getSuggestedName(resourceType, target);
    } else if (suggestedName.ends_with('/')) {
        _name = suggestedName + getSuggestedNameWithoutFolder(resourceType);
    } else {
        _name = suggestedName;
    }

    // A new name starts with the description of what is stored under the suggested name, or else with the last description of its folder
    auto rememberedDescription = _descriptionByTypeAndFolder.find({_resourceType, getFolder(_name)});
    resetDescription(rememberedDescription != _descriptionByTypeAndFolder.end() ? rememberedDescription->second : std::string());
    updateDescription(evaluate());
    if (_descriptionEditable) {
        _ownDescription = _description;
    }

    if (_resourceType == NetworkResourceType_Simulation) {
        changeTitle("Save simulation");
        _preview.createForSimulation();
    } else {
        changeTitle("Save genome");
        _preview.createForGenome(GenomeEditorWindow::get().getCurrentPreviewDescs());
    }
    AlienDialog::open();
}

void SaveSimulationDialog::openForSharing(
    NetworkResourceType resourceType,
    std::filesystem::path const& file,
    std::string const& name,
    std::string const& description)
{
    if (!NetworkService::get().getLoggedInUserName()) {
        LoginDialog::get().open();
        return;
    }
    _simulationFile.reset();
    _savedGenome.reset();
    _replacedUpload.reset();
    if (resourceType == NetworkResourceType_Simulation) {
        _simulationFile = file;
    } else {
        GenomeDesc genome;
        if (!SerializerService::get().deserializeGenomeFromFile(genome, file)) {
            showMessage("Share genome", "The selected file could not be opened.");
            return;
        }
        _savedGenome = genome;
    }
    _mode = Mode::Sharing;
    _resourceType = resourceType;
    _targets = {SaveTarget::Community, SaveTarget::Private};
    _target = SaveTarget::Community;
    _errorMessage.reset();
    _name = name;
    resetDescription(description);

    changeTitle(resourceType == NetworkResourceType_Simulation ? "Share simulation" : "Share genome");
    _preview.createForPicture(SimulationLibraryService::get().loadPicture(file).value_or(std::string()));
    AlienDialog::open();
}

void SaveSimulationDialog::openForReplacing(NetworkResourceRawTO const& upload)
{
    if (!NetworkService::get().getLoggedInUserName()) {
        LoginDialog::get().open();
        return;
    }
    _mode = Mode::Replacing;
    _resourceType = upload->resourceType;
    _target = upload->workspaceType == WorkspaceType_Private ? SaveTarget::Private : SaveTarget::Community;
    _targets = {_target};
    _simulationFile.reset();
    _savedGenome.reset();
    _replacedUpload = upload;
    _errorMessage.reset();
    _name = upload->resourceName;
    resetDescription(upload->description);

    if (_resourceType == NetworkResourceType_Simulation) {
        changeTitle("Replace simulation");
        _preview.createForSimulation();
    } else {
        changeTitle("Replace genome");
        _preview.createForGenome(GenomeEditorWindow::get().getCurrentPreviewDescs());
    }
    AlienDialog::open();
}

SaveSimulationDialog::SaveSimulationDialog()
    : AlienDialog("", {960.0f, 420.0f})
{}

void SaveSimulationDialog::processIntern()
{
    auto columnHeight = std::max(scale(MinColumnHeight), ImGui::GetContentRegionAvail().y - scale(FooterHeight));
    if (ImGui::BeginChild("##picture", {scale(PreviewWidth), columnHeight}, 0, ImGuiWindowFlags_NoScrollbar)) {
        auto pictureLabel = [&]() -> std::string {
            if (_mode == Mode::Sharing) {
                return ICON_FA_IMAGE "  Saved picture";
            }
            return _resourceType == NetworkResourceType_Simulation ? ICON_FA_CAMERA "  Current view" : ICON_FA_CAMERA "  Current genome";
        }();
        _preview.process(ImGui::GetContentRegionAvail().x, pictureLabel);
    }
    ImGui::EndChild();

    ImGui::SameLine(0, scale(ColumnSpacing));
    if (ImGui::BeginChild("##form", {0, columnHeight})) {
        processForm();
    }
    ImGui::EndChild();

    AlienGui::Separator();
    auto outcome = evaluate();
    updateDescription(outcome);
    processFooter(outcome);
}

void SaveSimulationDialog::processForm()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);

    // The file dialog asks for the name of a file in another location, which has no description. A replaced upload keeps both.
    auto ignoresNameAndDescription = _target == SaveTarget::OtherLocation || _mode == Mode::Replacing;

    processLabel("Name");
    ImGui::BeginDisabled(ignoresNameAndDescription);
    if (AlienGui::InputText(
            AlienGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType) + " name, use / to create folders").textWidth(0), _name)) {
        _errorMessage.reset();
    }
    ImGui::EndDisabled();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(
        "Use the `/`-notation to put the " + resourceTypeString
        + " into a folder. Missing folders are created automatically.\nFor instance, the name `Biome/Water world/Initial/Variant 1` creates the nested "
          "folders `Biome`, `Water world` and `Initial`."));

    addSectionSpacing();
    processLabel("Description", "optional");
    auto hasTargetSelection = _targets.size() > 1;
    auto lowerSectionHeight = _mode == Mode::Replacing ? ReplaceSectionHeight : (hasTargetSelection ? TargetSectionHeight : 0.0f);
    ImGui::PushID("description");
    ImGui::BeginGroup();
    ImGui::BeginDisabled(!_descriptionEditable || ignoresNameAndDescription);
    if (AlienGui::InputTextMultiline(
            AlienGui::InputTextMultilineParameters()
                .hint(
                    _resourceType == NetworkResourceType_Simulation ? "What happens in it? What should people watch for?"
                                                                    : "What does the creature do? What should people watch for?")
                .textWidth(0)
                .height(std::max(MinDescriptionHeight, scaleInverse(ImGui::GetContentRegionAvail().y) - lowerSectionHeight)),
            _description)) {
        _descriptionEdited = true;
        _ownDescription = _description;
    }
    ImGui::EndDisabled();
    ImGui::EndGroup();
    ImGui::PopID();
    if (!_descriptionEditable) {
        AlienGui::Tooltip(AlienGui::TooltipParameters().text(
            "The description of the simulation with this name is longer than this field can show. It is kept and can be changed in its text file."));
    }
    auto privacyTooltip =
        "The " + resourceTypeString + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.";
    if (_mode == Mode::Replacing) {
        auto replacement = std::string(_resourceType == NetworkResourceType_Simulation ? "the current simulation" : "the genome of the genome editor");
        addSectionSpacing();
        ImGui::PushTextWrapPos(0.0f);
        AlienGui::Text("The content and the picture of this upload are replaced by " + replacement + ". Its name, description and reactions are kept.");
        ImGui::PopTextWrapPos();
        AlienGui::PrivacyPolicyLink(privacyTooltip);
        return;
    }
    if (!hasTargetSelection) {
        return;
    }

    addSectionSpacing();
    auto choosesPlace = _mode == Mode::RunningResource;
    processLabel(choosesPlace ? "Save to" : "Share with");
    std::vector<std::string> targetLabels;
    for (auto const& target : _targets) {
        targetLabels.emplace_back(getTargetLabel(target, _resourceType, choosesPlace));
    }
    auto targetIndex = toInt(std::distance(_targets.begin(), std::ranges::find(_targets, _target)));
    if (AlienGui::SegmentedControl(AlienGui::SegmentedControlParameters().values(targetLabels), targetIndex)) {
        _target = _targets.at(targetIndex);
        _errorMessage.reset();
    }

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    AlienGui::Text(getTargetExplanation(_target, resourceTypeString));
    ImGui::PopTextWrapPos();
    if (isServerTarget(_target)) {
        AlienGui::PrivacyPolicyLink(privacyTooltip);
    }
}

void SaveSimulationDialog::processFooter(Outcome const& outcome)
{
    auto buttonText = [&]() -> std::string {
        if (outcome.needsLogin) {
            return ICON_FA_SIGN_IN_ALT "  Log in";
        }
        if (_target == SaveTarget::OtherLocation) {
            return ICON_FA_SAVE "  Save...";
        }
        if (_target == SaveTarget::ThisComputer) {
            return outcome.replacesFile ? ICON_FA_SAVE "  Replace" : ICON_FA_SAVE "  Save";
        }
        if (outcome.replacedResourceId.has_value()) {
            return ICON_FA_EXCHANGE_ALT "  Replace";
        }
        return _target == SaveTarget::Community ? ICON_FA_UPLOAD "  Share with the community" : ICON_FA_LOCK "  Upload privately";
    }();
    ImGui::BeginDisabled(!outcome.possible);
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(buttonText).highlighted(true).frame(true).transparentBackground(false))) {
        onConfirm(outcome);
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }

    if (auto const& message = _errorMessage.has_value() ? _errorMessage : outcome.message) {
        ImGui::SameLine();
        auto messageWidth = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x;
        if (messageWidth >= scale(MinMessageWidth)) {
            if (ImGui::BeginChild("##message", {messageWidth, ImGui::GetFrameHeight()}, 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                ImGui::AlignTextToFramePadding();
                ImGui::PushStyleColor(ImGuiCol_Text, Const::WarningColor.Value);
                AlienGui::Text(AlienGui::TextParameters().text(*message).truncate(true));
                ImGui::PopStyleColor();
            }
            ImGui::EndChild();
            AlienGui::Tooltip(AlienGui::TooltipParameters().text(*message));
        }
    }
}

auto SaveSimulationDialog::evaluate() const -> Outcome
{
    Outcome result;
    auto name = boost::algorithm::trim_copy(_name);

    if (_mode == Mode::Replacing) {
        result.possible = true;
        result.replacedResourceId = _replacedUpload->id;
        return result;
    }
    if (_target == SaveTarget::OtherLocation) {
        result.possible = true;
        return result;
    }
    if (_target == SaveTarget::ThisComputer) {
        auto& service = SimulationLibraryService::get();
        if (name.empty() || name.ends_with('/')) {
            return result;
        }
        auto isGenome = _resourceType == NetworkResourceType_Genome;
        if (!(isGenome ? service.isValidGenomeName(name) : service.isValidSimulationName(name))) {
            result.message = !isGenome && service.isValidFolderName(name)
                ? "Names of this form are reserved for autosaves"
                : "A name must not contain < > : \" \\ | ? * and must not begin or end with a dot or a space";
            return result;
        }
        auto const& libraryFolder = SimulationLibraryController::get().getLibraryFolder();
        result.replacesFile = isGenome ? service.isGenomeNameTaken(libraryFolder, name) : service.isSimulationNameTaken(libraryFolder, name);
        if (result.replacesFile) {
            auto file = findLibraryFile(_resourceType, name);
            result.message = file != nullptr
                ? "Replaces the " + BrowserDataTypeToLowerString.at(_resourceType) + " saved on " + file->timestamp.substr(0, TimestampLengthWithoutSeconds)
                : std::string("Replaces a file with this name");
        }
        result.possible = true;
        return result;
    }

    if (!NetworkService::get().getLoggedInUserName().has_value()) {
        result.possible = true;
        result.needsLogin = true;
        result.message = "Uploading needs an account";
        return result;
    }
    if (NetworkResourceService::get().removeFoldersFromName(name).empty()) {
        return result;
    }
    if (_mode == Mode::RunningResource) {
        if (auto rawTO = findOwnUpload(_resourceType, _target, name)) {
            result.replacedResourceId = rawTO->id;
            result.message = "Replaces your upload of " + rawTO->timestamp.substr(0, DateLength) + " and keeps its reactions";
        }
    }
    result.possible = true;
    return result;
}

void SaveSimulationDialog::resetDescription(std::string const& description)
{
    _description = description;
    _ownDescription = description;
    _descriptionEdited = false;
    _descriptionEditable = true;
    _descriptionSource.reset();
}

// The description field shows what is stored under the name as long as nothing has been written into it, so that replacing a simulation or
// an upload does not replace its description by the description of something else
void SaveSimulationDialog::updateDescription(Outcome const& outcome)
{
    if (_mode != Mode::RunningResource) {
        return;
    }
    DescriptionSource source{
        .target = _target, .name = boost::algorithm::trim_copy(_name), .replacesFile = outcome.replacesFile, .replacedResourceId = outcome.replacedResourceId};
    if (_descriptionSource == source) {
        return;
    }
    _descriptionSource = source;

    // A description that is longer than the field can show cannot be written here
    auto replacedDescription = findReplacedDescription(outcome);
    _descriptionEditable = !replacedDescription.has_value() || replacedDescription->complete;
    if (replacedDescription.has_value() && (!_descriptionEdited || !_descriptionEditable)) {
        _description = replacedDescription->text;
    } else {
        _description = _ownDescription;
    }
}

std::optional<SimulationDescription> SaveSimulationDialog::findReplacedDescription(Outcome const& outcome) const
{
    auto name = boost::algorithm::trim_copy(_name);
    if (outcome.replacesFile) {

        // Read from the file, since the name may be spelled differently than the simulation it replaces
        auto& service = SimulationLibraryService::get();
        auto const& libraryFolder = SimulationLibraryController::get().getLibraryFolder();
        return service.loadDescription(
            _resourceType == NetworkResourceType_Simulation ? service.getSimulationPath(libraryFolder, name) : service.getGenomePath(libraryFolder, name));
    }
    if (outcome.replacedResourceId.has_value()) {
        if (auto rawTO = findOwnUpload(_resourceType, _target, name)) {
            return SimulationDescription{.text = rawTO->description};
        }
    }
    return std::nullopt;
}

void SaveSimulationDialog::onConfirm(Outcome const& outcome)
{
    auto name = boost::algorithm::trim_copy(_name);
    if (outcome.needsLogin) {
        close();
        LoginDialog::get().open();
        return;
    }

    if (_mode == Mode::Replacing) {
        close();
        onReplaceUpload(_replacedUpload->id, _replacedUpload->workspaceType);
        return;
    }
    if (_target == SaveTarget::OtherLocation) {
        close();

        // Executed after this dialog has been closed
        delayedExecution([resourceType = _resourceType] {
            if (resourceType == NetworkResourceType_Simulation) {
                FileTransferController::get().onSaveSimulationDialog();
            } else {
                FileTransferController::get().onSaveGenomeDialog(
                    GenomeEditorWindow::get().getCurrentGenome(), [] { GenomeEditorWindow::get().markCurrentGenomeAsSaved(); });
            }
        });
        return;
    }
    if (_target == SaveTarget::ThisComputer) {
        auto& controller = SimulationLibraryController::get();

        // What is replaced keeps its description unless another one has been written here
        auto keepsDescription = outcome.replacesFile && (!_descriptionEdited || !_descriptionEditable);
        if (_descriptionEditable) {
            _descriptionByTypeAndFolder[{_resourceType, getFolder(name)}] = _description;
        }
        auto description = keepsDescription ? std::nullopt : std::make_optional(_description);
        if (_resourceType == NetworkResourceType_Simulation) {
            controller.onSave(name, description, _preview.getJpg());
        } else {
            if (auto errorMessage = controller.onSaveGenome(name, GenomeEditorWindow::get().getCurrentGenome(), description, _preview.getJpg())) {
                _errorMessage = errorMessage;
                return;
            }
            GenomeEditorWindow::get().markCurrentGenomeAsSaved();
        }
        close();
        return;
    }

    if (!NetworkValidationService::get().isStringValidForDatabase(name) || !NetworkValidationService::get().isStringValidForDatabase(_description)) {
        showMessage("Error", Const::NotAllowedCharacters);
        return;
    }
    if (_mode == Mode::RunningResource) {
        _descriptionByTypeAndFolder[{_resourceType, getFolder(name)}] = _description;
    }
    close();
    if (outcome.replacedResourceId.has_value()) {
        onReplaceUpload(*outcome.replacedResourceId, toWorkspaceType(_target));
    } else {
        onUpload();
    }
}

void SaveSimulationDialog::onUpload()
{
    auto data = [&]() -> std::variant<UploadNetworkResourceRequestData::SimulationData, UploadNetworkResourceRequestData::CreatureData> {
        if (_resourceType == NetworkResourceType_Simulation) {
            return UploadNetworkResourceRequestData::SimulationData{
                .zoom = Viewport::get().getZoomFactor(),
                .center = Viewport::get().getCenterInWorldPos(),
                .jpg = _preview.getJpg(),
                .filename = _simulationFile};
        } else {
            return UploadNetworkResourceRequestData::CreatureData{
                .description = _savedGenome.has_value() ? *_savedGenome : GenomeEditorWindow::get().getCurrentGenome(), .jpg = _preview.getJpg()};
        }
    }();
    NetworkTransferController::get().onUpload(UploadNetworkResourceRequestData{
        .resourceWithoutFolderName = boost::algorithm::trim_copy(_name),
        .resourceDescription = _description,
        .workspaceType = toWorkspaceType(_target),
        .downloadCache = BrowserWindow::get().getSimulationCache(),
        .data = data});
}

void SaveSimulationDialog::onReplaceUpload(std::string const& resourceId, WorkspaceType workspaceType)
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
        .resourceId = resourceId, .workspaceType = workspaceType, .downloadCache = BrowserWindow::get().getSimulationCache(), .data = data});

    // Replacing keeps the name and the description, so a description written in this dialog is stored separately
    auto name = boost::algorithm::trim_copy(_name);
    auto rawTO = findOwnUpload(_resourceType, _target, name);
    if (_descriptionEdited && rawTO != nullptr && rawTO->description != _description) {
        EditNetworkResourceRequestData::Entry entry{.resourceId = resourceId, .newName = name, .newDescription = _description};
        NetworkTransferController::get().onEdit(EditNetworkResourceRequestData{.entries = std::vector{entry}});
    }
}
