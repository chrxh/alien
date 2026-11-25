#include "SaveSimulationDialog.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <ranges>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/trim.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>
#include <Network/NetworkValidationService.h>

#include <Persister/Interface/LocalLibraryService.h>
#include <Persister/Interface/SerializerService.h>

#include "AlienGui.h"
#include "BrowserController.h"
#include "BrowserData.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "FileTransferController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LocalLibraryController.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "SimulationSessionGuiService.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    // The dialog keeps one size under all its titles
    auto constexpr WindowId = "###SaveSimulationDialog";

    auto constexpr PreviewWidth = 450.0f;
    auto constexpr ColumnSpacing = 24.0f;
    auto constexpr FooterHeight = 50.0f;
    auto constexpr MinColumnHeight = 100.0f;
    auto constexpr MinDescriptionHeight = 50.0f;
    auto constexpr SectionSpacing = 10.0f;
    auto constexpr TargetSectionHeight = 164.0f;
    auto constexpr ReplaceSectionHeight = 90.0f;
    auto constexpr AnnouncementOptionHeight = 26.0f;
    auto constexpr TimestampLengthWithoutSeconds = 16;  // e.g. 2026-11-13 14:02
    auto constexpr DateLength = 10;

    std::map<NetworkResourceType, std::string> const BrowserDataTypeToLowerString = {
        {NetworkResourceType_Simulation, "simulation"},
        {NetworkResourceType_Genome, "genome"}};
    std::map<NetworkResourceType, std::string> const BrowserDataTypeToUpperString = {
        {NetworkResourceType_Simulation, "Simulation"},
        {NetworkResourceType_Genome, "Genome"}};

    std::map<NetworkResourceType, std::string> const LastTargetSettingsKeyByType = {
        {NetworkResourceType_Simulation, "dialogs.save simulation.target"},
        {NetworkResourceType_Genome, "dialogs.save genome.target"}};
    std::map<SaveTarget, std::string> const SettingsValueByTarget = {
        {SaveTarget::ThisComputer, "this computer"},
        {SaveTarget::Community, "community"},
        {SaveTarget::Private, "private"},
        {SaveTarget::OtherLocation, "other location"}};
    auto constexpr AnnounceOnDiscordSettingsKey = "dialogs.replace upload.announce on discord";

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

    bool isServerTarget(SaveTarget target)
    {
        return target == SaveTarget::Community || target == SaveTarget::Private;
    }

    // Saved simulations and genomes are collected in a folder with the name of the user
    std::string getNameInUserFolder(std::string const& name)
    {
        auto userName = NetworkService::get().getLoggedInUserName();
        if (!userName.has_value()) {
            return name;
        }
        auto userFolder = *userName + "/";
        return name.starts_with(userFolder) ? name : userFolder + name;
    }

    // A simulation of this computer keeps its name
    std::string getSuggestedName(NetworkResourceType resourceType, SaveTarget target)
    {
        auto name = getNameInUserFolder(getSuggestedNameWithoutFolder(resourceType));
        if (!isServerTarget(target) && resourceType == NetworkResourceType_Simulation) {
            return SimulationSessionGuiService::get().getNameInLibrary().value_or(name);
        }
        return name;
    }

    std::string getFolder(std::string const& name)
    {
        auto& resourceService = NetworkResourceService::get();
        return resourceService.concatenateFolderName(resourceService.getFolderNames(name), true);
    }

    WorkspaceType toWorkspaceType(SaveTarget target)
    {
        return target == SaveTarget::Private ? WorkspaceType_Private : WorkspaceType_Community;
    }

    SaveTarget toSaveTarget(WorkspaceType workspaceType)
    {
        if (isLocalWorkspace(workspaceType)) {
            return SaveTarget::ThisComputer;
        }
        return workspaceType == WorkspaceType_Private ? SaveTarget::Private : SaveTarget::Community;
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

    std::string getTargetExplanation(SaveTarget target, std::string const& resourceTypeString, bool isNewUpload)
    {
        switch (target) {
        case SaveTarget::ThisComputer:
            return "The " + resourceTypeString + " is stored on this computer. You can share it from the browser at any time.";
        case SaveTarget::Community:
            return "Everyone can see and download the " + resourceTypeString + " in the Community workspace. "
                + (isNewUpload ? "It is also announced on the ALIEN Discord. " : "") + "Work in progress is welcome.";
        case SaveTarget::OtherLocation:
            return "The " + resourceTypeString + " is stored as a file in any folder. Save asks for the folder and the file name.";
        default:
            return "The " + resourceTypeString
                + " is stored on the server, where only you can see it. It appears under My uploads in the browser and can be shared later.";
        }
    }

    LibraryFile const* findLibraryFile(NetworkResourceType resourceType, std::string const& name)
    {
        auto const& content = LocalLibraryController::get().getContent();
        auto const& files = resourceType == NetworkResourceType_Simulation ? content.simulations : content.genomes;
        auto findResult = std::ranges::find_if(files, [&](LibraryFile const& file) { return file.name == name; });
        return findResult != files.end() ? &*findResult : nullptr;
    }

    // A name belongs to one upload of the user, whoever can see it. Of several uploads with the name, the one with the visibility of the target
    // and then the newest one is replaced.
    NetworkResourceRawTO findOwnUpload(NetworkResourceType resourceType, SaveTarget target, std::string const& name)
    {
        auto uploads = BrowserController::get().getData()->findOwnUploads(resourceType, name);
        if (uploads.empty()) {
            return nullptr;
        }
        return std::ranges::max(uploads, {}, [&](NetworkResourceRawTO const& upload) {
            return std::make_tuple(upload->workspaceType == toWorkspaceType(target), upload->timestamp, upload->id);
        });
    }

    // Replacing keeps the visibility of an upload, which may differ from the target
    std::string getReplacementMessage(NetworkResourceRawTO const& upload, SaveTarget target)
    {
        auto date = upload->timestamp.substr(0, DateLength);
        if (upload->workspaceType == toWorkspaceType(target)) {
            return "Replaces your upload of " + date + " and keeps its reactions";
        }
        auto place = [&] {
            switch (upload->workspaceType) {
            case WorkspaceType_Private:
                return "My uploads";
            case WorkspaceType_Featured:
                return "Featured";
            default:
                return "Community";
            }
        }();
        return "Replaces your upload of " + date + " in " + place + ", which keeps its visibility and reactions";
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

    // The genome of the genome editor counts as saved once the server has stored it, unless it has been changed in the meantime
    std::function<void()> createMarkGenomeAsSavedFunc()
    {
        return [genome = GenomeEditorWindow::get().getCurrentGenome()] {
            auto& genomeEditor = GenomeEditorWindow::get();
            if (genomeEditor.getCurrentGenome() == genome) {
                genomeEditor.markCurrentGenomeAsSaved();
            }
        };
    }
}

void SaveSimulationDialog::open(
    NetworkResourceType resourceType,
    std::optional<SaveTarget> target,
    std::string const& suggestedName,
    std::optional<SaveTarget> suggestedTarget)
{
    auto isLoggedIn = NetworkService::get().getLoggedInUserName().has_value();

    // A plain save never asks for a login, so a suggested or remembered upload falls back to this computer
    auto remembersTarget = !target.has_value();
    if (remembersTarget) {
        target = suggestedTarget.value_or(_lastTargetByType.at(resourceType));
        if (isServerTarget(*target) && !isLoggedIn) {
            target = SaveTarget::ThisComputer;
        }
    }
    if (isServerTarget(*target) && !isLoggedIn) {
        LoginDialog::get().open();
        return;
    }
    _mode = Mode::RunningResource;
    _resourceType = resourceType;
    _targets = {SaveTarget::ThisComputer, SaveTarget::Community, SaveTarget::Private, SaveTarget::OtherLocation};
    _target = *target;
    _remembersTarget = remembersTarget;
    _targetChosen = false;
    _simulationFile.reset();
    _savedGenome.reset();
    _replacedResource.reset();
    _errorMessage.reset();

    if (suggestedName.empty()) {
        _name = getSuggestedName(resourceType, _target);
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
        changeTitle("Save simulation" + std::string(WindowId));
        _preview.createForSimulation();
    } else {
        changeTitle("Save genome" + std::string(WindowId));
        _preview.createForGenome(GenomeEditorWindow::get().getCurrentPreviewDescs());
    }
    _uploadsRefreshed = false;
    AlienDialog::open();
}

void SaveSimulationDialog::openForSharing(
    NetworkResourceType resourceType,
    std::filesystem::path const& file,
    std::string const& nameInLibrary,
    std::string const& description)
{
    if (!NetworkService::get().getLoggedInUserName()) {
        LoginDialog::get().open();
        return;
    }
    _simulationFile.reset();
    _savedGenome.reset();
    _replacedResource.reset();
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
    _name = getNameInUserFolder(nameInLibrary);
    resetDescription(description);

    changeTitle(std::string(resourceType == NetworkResourceType_Simulation ? "Share simulation" : "Share genome") + WindowId);
    _preview.createForPicture(LocalLibraryService::get().loadPicture(file).value_or(std::string()));
    _uploadsRefreshed = false;
    AlienDialog::open();
}

void SaveSimulationDialog::openForReplacing(NetworkResourceRawTO const& resource)
{
    auto target = toSaveTarget(resource->workspaceType);
    if (isServerTarget(target) && !NetworkService::get().getLoggedInUserName()) {
        LoginDialog::get().open();
        return;
    }
    _mode = Mode::Replacing;
    _resourceType = resource->resourceType;
    _target = target;
    _targets = {_target};
    _simulationFile.reset();
    _savedGenome.reset();
    _replacedResource = resource;
    _errorMessage.reset();
    _name = resource->resourceName;
    resetDescription(resource->description);

    if (_resourceType == NetworkResourceType_Simulation) {
        changeTitle("Replace simulation" + std::string(WindowId));
        _preview.createForSimulation();
    } else {
        changeTitle("Replace genome" + std::string(WindowId));
        _preview.createForGenome(GenomeEditorWindow::get().getCurrentPreviewDescs());
    }
    AlienDialog::open();
}

SaveSimulationDialog::SaveSimulationDialog()
    : AlienDialog("", {960.0f, 420.0f})
{}

void SaveSimulationDialog::initIntern()
{
    auto& settings = GlobalSettings::get();
    for (auto const& [resourceType, key] : LastTargetSettingsKeyByType) {
        auto value = settings.getValue(key, SettingsValueByTarget.at(SaveTarget::ThisComputer));
        auto findResult = std::ranges::find_if(SettingsValueByTarget, [&](auto const& entry) { return entry.second == value; });
        _lastTargetByType[resourceType] = findResult != SettingsValueByTarget.end() ? findResult->first : SaveTarget::ThisComputer;
    }
    _announceOnDiscord = settings.getValue(AnnounceOnDiscordSettingsKey, _announceOnDiscord);
}

void SaveSimulationDialog::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    for (auto const& [resourceType, target] : _lastTargetByType) {
        settings.setValue(LastTargetSettingsKeyByType.at(resourceType), SettingsValueByTarget.at(target));
    }
    settings.setValue(AnnounceOnDiscordSettingsKey, _announceOnDiscord);
}

void SaveSimulationDialog::processIntern()
{
    // An upload with the name may come from elsewhere, so the uploads are refreshed once a target on the server is chosen
    if (isServerTarget(_target) && !_uploadsRefreshed && NetworkService::get().getLoggedInUserName().has_value()) {
        _uploadsRefreshed = true;
        if (!BrowserController::get().isRefreshing()) {
            BrowserController::get().refresh(false);
        }
    }

    auto outcome = evaluate();
    updateDescription(outcome);

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
        processForm(outcome);
    }
    ImGui::EndChild();

    AlienGui::Separator();
    processFooter(outcome);
}

void SaveSimulationDialog::processForm(Outcome const& outcome)
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);

    // The file dialog asks for the name of a file in another location, which has no description. What is replaced keeps both.
    auto ignoresNameAndDescription = _target == SaveTarget::OtherLocation || _mode == Mode::Replacing;

    processLabel("Name");
    if (!ignoresNameAndDescription) {
        AlienGui::HelpMarker(
            "Use the `/`-notation to put the " + resourceTypeString
            + " into a folder. Missing folders are created automatically.\nFor instance, the name `Biome/Water world/Initial/Variant 1` creates the nested "
              "folders `Biome`, `Water world` and `Initial`.");
    }
    ImGui::BeginDisabled(ignoresNameAndDescription);
    if (AlienGui::InputText(
            AlienGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType) + " name, use / to create folders").textWidth(0), _name)) {
        _errorMessage.reset();
    }
    ImGui::EndDisabled();
    if (auto const& message = _errorMessage.has_value() ? _errorMessage : outcome.message) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::WarningColor.Value);
        ImGui::PushTextWrapPos(0.0f);
        AlienGui::Text(*message);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }

    addSectionSpacing();
    if (_descriptionEditable) {
        processLabel("Description", "optional");
    } else {
        processLabel("Description");
        AlienGui::HelpMarker(
            "The description of the " + resourceTypeString
            + " with this name is longer than this field can show. It is kept and can be changed in its text file.");
    }
    auto hasTargetSelection = _targets.size() > 1;
    auto lowerSectionHeight = [&] {
        if (_mode == Mode::Replacing) {
            return outcome.replacesSharedUpload ? ReplaceSectionHeight + AnnouncementOptionHeight : ReplaceSectionHeight;
        }
        return hasTargetSelection ? TargetSectionHeight : 0.0f;
    }();
    ImGui::PushID("description");
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
    ImGui::PopID();
    auto privacyTooltip =
        "The " + resourceTypeString + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.";
    if (_mode == Mode::Replacing) {
        auto replacement = std::string(_resourceType == NetworkResourceType_Simulation ? "the current simulation" : "the genome of the genome editor");
        auto explanation = isServerTarget(_target)
            ? "The content and the picture of this upload are replaced by " + replacement + ". Its name, description and reactions are kept."
            : "The content and the picture of this " + resourceTypeString + " are replaced by " + replacement + ". Its name and description are kept.";
        addSectionSpacing();
        ImGui::PushTextWrapPos(0.0f);
        AlienGui::Text(explanation);
        ImGui::PopTextWrapPos();
        processAnnouncementOption(outcome);
        if (isServerTarget(_target)) {
            AlienGui::PrivacyPolicyLink(privacyTooltip);
        }
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
    if (AlienGui::ChoiceBar(AlienGui::ChoiceBarParameters().values(targetLabels), targetIndex)) {
        _target = _targets.at(targetIndex);
        _targetChosen = true;
        _errorMessage.reset();
    }

    auto isNewUpload = !outcome.replacedResourceId.has_value();
    auto explainedTarget = _target;
    if (!isNewUpload) {
        explainedTarget = outcome.replacesSharedUpload ? SaveTarget::Community : SaveTarget::Private;
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    AlienGui::Text(getTargetExplanation(explainedTarget, resourceTypeString, isNewUpload));
    ImGui::PopTextWrapPos();
    processAnnouncementOption(outcome);
    if (isServerTarget(_target)) {
        AlienGui::PrivacyPolicyLink(privacyTooltip);
    }
}

void SaveSimulationDialog::processAnnouncementOption(Outcome const& outcome)
{
    if (!outcome.replacesSharedUpload) {
        return;
    }
    ImGui::Checkbox("##announceOnDiscord", &_announceOnDiscord);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    AlienGui::Text("Announce on the ALIEN Discord");
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
        if (_mode == Mode::Replacing || outcome.replacedResourceId.has_value()) {
            return ICON_FA_EXCHANGE_ALT "  Replace";
        }
        if (_target == SaveTarget::ThisComputer) {
            return outcome.replacesFile ? ICON_FA_SAVE "  Replace" : ICON_FA_SAVE "  Save";
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
}

auto SaveSimulationDialog::evaluate() const -> Outcome
{
    Outcome result;
    auto name = boost::algorithm::trim_copy(_name);

    if (_mode == Mode::Replacing) {
        result.possible = true;
        if (isServerTarget(_target)) {
            result.replacedResourceId = _replacedResource->id;
            result.replacesSharedUpload = _replacedResource->workspaceType != WorkspaceType_Private;
        } else {
            result.replacesFile = true;
        }
        return result;
    }
    if (_target == SaveTarget::OtherLocation) {
        result.possible = true;
        return result;
    }
    if (_target == SaveTarget::ThisComputer) {
        auto& service = LocalLibraryService::get();
        if (name.empty() || name.ends_with('/')) {
            return result;
        }
        auto isGenome = _resourceType == NetworkResourceType_Genome;
        if (!(isGenome ? service.isValidGenomeName(name) : service.isValidSimulationName(name))) {
            result.message = !isGenome && service.isValidFolderName(name)
                ? "The name of a simulation must not end with .genome"
                : "A name must not contain < > : \" \\ | ? * and must not begin or end with a dot or a space";
            return result;
        }
        auto const& libraryFolder = LocalLibraryController::get().getLibraryFolder();
        result.replacesFile = isGenome ? service.isGenomeNameTaken(libraryFolder, name) : service.isSimulationNameTaken(libraryFolder, name);
        if (result.replacesFile) {
            auto file = findLibraryFile(_resourceType, name);
            result.message = file != nullptr
                ? "The file already exists (saved " + file->timestamp.substr(0, TimestampLengthWithoutSeconds) + ") and will be replaced."
                : std::string("A file with this name already exists and will be replaced.");
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

    if (BrowserController::get().isRefreshing() || NetworkTransferController::get().hasPendingNameChanges()) {
        result.message = "Looking for an upload with this name ...";
        return result;
    }
    if (auto rawTO = findOwnUpload(_resourceType, _target, name)) {
        result.replacedResourceId = rawTO->id;
        result.replacesSharedUpload = rawTO->workspaceType != WorkspaceType_Private;
        result.message = getReplacementMessage(rawTO, _target);
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

std::optional<FileDescription> SaveSimulationDialog::findReplacedDescription(Outcome const& outcome) const
{
    auto name = boost::algorithm::trim_copy(_name);
    if (outcome.replacesFile) {

        // Read from the file, since the name may be spelled differently than the simulation it replaces
        auto& service = LocalLibraryService::get();
        auto const& libraryFolder = LocalLibraryController::get().getLibraryFolder();
        return service.loadDescription(
            _resourceType == NetworkResourceType_Simulation ? service.getSimulationPath(libraryFolder, name) : service.getGenomePath(libraryFolder, name));
    }
    if (outcome.replacedResourceId.has_value()) {
        if (auto rawTO = findOwnUpload(_resourceType, _target, name)) {
            return FileDescription{.text = rawTO->description};
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
        if (_target == SaveTarget::ThisComputer) {

            // What is replaced keeps its name and its description
            if (_resourceType == NetworkResourceType_Simulation) {
                LocalLibraryController::get().onReplace(StringHelper::toPath(_replacedResource->id), _preview.getJpg());
            } else if (auto errorMessage = saveOnThisComputer(_replacedResource->resourceName, std::nullopt)) {
                _errorMessage = errorMessage;
                return;
            }
            close();
            return;
        }
        close();
        onReplaceUpload(outcome, _replacedResource->workspaceType);
        return;
    }
    if (_mode == Mode::RunningResource && _remembersTarget && _targetChosen) {
        _lastTargetByType[_resourceType] = _target;
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

        // What is replaced keeps its description unless another one has been written here
        auto keepsDescription = outcome.replacesFile && (!_descriptionEdited || !_descriptionEditable);
        if (_descriptionEditable) {
            _descriptionByTypeAndFolder[{_resourceType, getFolder(name)}] = _description;
        }
        auto description = keepsDescription ? std::nullopt : std::make_optional(_description);
        if (auto errorMessage = saveOnThisComputer(name, description)) {
            _errorMessage = errorMessage;
            return;
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
        onReplaceUpload(outcome, toWorkspaceType(_target));
    } else {
        onUpload();
    }
}

std::optional<std::string> SaveSimulationDialog::saveOnThisComputer(std::string const& name, std::optional<std::string> const& description)
{
    auto& controller = LocalLibraryController::get();
    if (_resourceType == NetworkResourceType_Simulation) {
        controller.onSave(name, description, _preview.getJpg());
        return std::nullopt;
    }
    if (auto errorMessage = controller.onSaveGenome(name, GenomeEditorWindow::get().getCurrentGenome(), description, _preview.getJpg())) {
        return errorMessage;
    }
    GenomeEditorWindow::get().markCurrentGenomeAsSaved();
    return std::nullopt;
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
    auto isGenomeOfGenomeEditor = _resourceType == NetworkResourceType_Genome && !_savedGenome.has_value();
    NetworkTransferController::get().onUpload(
        UploadNetworkResourceRequestData{
            .resourceWithoutFolderName = boost::algorithm::trim_copy(_name),
            .resourceDescription = _description,
            .workspaceType = toWorkspaceType(_target),
            .downloadCache = BrowserWindow::get().getSimulationCache(),
            .data = data},
        isGenomeOfGenomeEditor ? createMarkGenomeAsSavedFunc() : std::function<void()>());
}

void SaveSimulationDialog::onReplaceUpload(Outcome const& outcome, WorkspaceType workspaceType)
{
    auto const& resourceId = *outcome.replacedResourceId;
    auto data = [&]() -> std::variant<ReplaceNetworkResourceRequestData::SimulationData, ReplaceNetworkResourceRequestData::CreatureData> {
        if (_resourceType == NetworkResourceType_Simulation) {
            return ReplaceNetworkResourceRequestData::SimulationData{
                .zoom = Viewport::get().getZoomFactor(),
                .center = Viewport::get().getCenterInWorldPos(),
                .jpg = _preview.getJpg(),
                .filename = _simulationFile};
        } else {
            return ReplaceNetworkResourceRequestData::CreatureData{
                .description = _savedGenome.has_value() ? *_savedGenome : GenomeEditorWindow::get().getCurrentGenome(), .jpg = _preview.getJpg()};
        }
    }();
    auto isGenomeOfGenomeEditor = _resourceType == NetworkResourceType_Genome && !_savedGenome.has_value();
    NetworkTransferController::get().onReplace(
        ReplaceNetworkResourceRequestData{
            .resourceId = resourceId,
            .workspaceType = workspaceType,
            .announceOnDiscord = _announceOnDiscord || !outcome.replacesSharedUpload,
            .downloadCache = BrowserWindow::get().getSimulationCache(),
            .data = data},
        isGenomeOfGenomeEditor ? createMarkGenomeAsSavedFunc() : std::function<void()>());

    // Replacing keeps the name and the description, so a description written in this dialog or that of a shared simulation is stored separately
    auto name = boost::algorithm::trim_copy(_name);
    auto rawTO = BrowserController::get().getData()->findRawTO(resourceId);
    if ((_descriptionEdited || _mode == Mode::Sharing) && rawTO != nullptr && rawTO->description != _description) {
        EditNetworkResourceRequestData::Entry entry{.resourceId = resourceId, .newName = name, .newDescription = _description};
        NetworkTransferController::get().onEdit(EditNetworkResourceRequestData{.entries = std::vector{entry}});
    }
}
