#include "SimulationLibraryService.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <ranges>
#include <set>

#include <Base/Definitions.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

namespace
{
    auto constexpr SimulationExtension = ".sim";
    auto constexpr GenomeExtension = ".genome";
    auto constexpr PictureExtension = ".jpg";
    auto constexpr DescriptionExtension = ".txt";
    auto constexpr TemporaryExtension = ".tmp";
    auto constexpr SavepointTableFilename = "savepoints.json";  // Written by the Autosave window of earlier versions
    auto constexpr NameSeparator = '/';
    auto constexpr Utf8ByteOrderMark = "\xef\xbb\xbf";
    auto constexpr MaxNameLength = 100;
    auto constexpr MaxNumFolders = 2000;
    auto constexpr MaxFolderDepth = 8;
    auto constexpr MaxNumFilesOfOneType = 20000;
    auto constexpr MaxNumTimestepDigits = 19;

    std::string const ForbiddenCharacters = "<>:\"/\\|?*";
    std::vector<std::string> const ReservedNames = {"CON",  "PRN",  "AUX",  "NUL",  "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
                                                    "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};

    // Names and extensions may contain UTF-8 sequences, which must stay untouched
    std::string toAsciiUpper(std::string const& text)
    {
        auto result = text;
        for (auto& ch : result) {
            if (ch >= 'a' && ch <= 'z') {
                ch = static_cast<char>(ch - 'a' + 'A');
            }
        }
        return result;
    }

    bool isReservedName(std::string const& name)
    {
        auto stem = toAsciiUpper(name.substr(0, name.find('.')));
        return std::ranges::find(ReservedNames, stem) != ReservedNames.end();
    }

    bool isForbiddenCharacter(char ch)
    {
        return static_cast<unsigned char>(ch) < 32 || ForbiddenCharacters.find(ch) != std::string::npos;
    }

    bool isUtf8ContinuationByte(char ch)
    {
        return (static_cast<unsigned char>(ch) & 0xc0) == 0x80;
    }

    bool isDigit(char ch)
    {
        return ch >= '0' && ch <= '9';
    }

    // The name of a file or of a folder without its path. What starts with a dot is not part of the library.
    bool isValidNamePart(std::string const& name)
    {
        if (name.empty() || name.size() > MaxNameLength || name.front() == ' ' || name.front() == '.' || name.back() == ' ' || name.back() == '.') {
            return false;
        }
        return std::ranges::none_of(name, isForbiddenCharacter) && !isReservedName(name);
    }

    std::vector<std::string> splitName(std::string const& name)
    {
        std::vector<std::string> result;
        size_t start = 0;
        while (true) {
            auto end = name.find(NameSeparator, start);
            if (end == std::string::npos) {
                result.emplace_back(name.substr(start));
                return result;
            }
            result.emplace_back(name.substr(start, end - start));
            start = end + 1;
        }
    }

    bool hasExtension(std::filesystem::path const& path, std::string const& extension)
    {
        return toAsciiUpper(StringHelper::toUtf8(path.extension())) == toAsciiUpper(extension);
    }

    // Links and junctions are neither followed nor deleted
    bool isRealDirectory(std::filesystem::directory_entry const& entry)
    {
        std::error_code error;
        return entry.symlink_status(error).type() == std::filesystem::file_type::directory && !error;
    }

    bool isRealFile(std::filesystem::directory_entry const& entry)
    {
        std::error_code error;
        return entry.symlink_status(error).type() == std::filesystem::file_type::regular && !error;
    }

    // Files and folders whose names start with a dot are not part of the library, like the companion files that macOS creates
    bool isHidden(std::filesystem::directory_entry const& entry)
    {
        return StringHelper::toUtf8(entry.path().filename()).starts_with('.');
    }

    bool isSimulationFile(std::filesystem::directory_entry const& entry)
    {
        return isRealFile(entry) && !isHidden(entry) && hasExtension(entry.path(), SimulationExtension);
    }

    bool isGenomeFile(std::filesystem::directory_entry const& entry)
    {
        return isRealFile(entry) && !isHidden(entry) && hasExtension(entry.path(), GenomeExtension);
    }

    bool isLibraryFolder(std::filesystem::directory_entry const& entry)
    {
        return isRealDirectory(entry) && !isHidden(entry);
    }

    // The iterator is advanced with an error code because a range-based loop throws on inaccessible entries
    template <typename Func>
    void forEachEntry(std::filesystem::path const& folder, Func const& func)
    {
        std::error_code iterationError;
        for (auto iterator = std::filesystem::directory_iterator(folder, iterationError); !iterationError && iterator != std::filesystem::directory_iterator();
             iterator.increment(iterationError)) {
            func(*iterator);
        }
    }

    // Reads the time step of a name like save_12_400_000 or save_12_400_000-2, where the postfix distinguishes saves of the same time step
    std::optional<uint64_t> parseTimestep(std::string const& name, std::string const& prefix)
    {
        if (!name.starts_with(prefix)) {
            return std::nullopt;
        }
        auto timestepPart = name.substr(prefix.size());
        if (auto postfixPos = timestepPart.find('-'); postfixPos != std::string::npos) {
            auto postfix = timestepPart.substr(postfixPos + 1);
            if (postfix.empty() || !std::ranges::all_of(postfix, isDigit)) {
                return std::nullopt;
            }
            timestepPart = timestepPart.substr(0, postfixPos);
        }

        uint64_t result = 0;
        auto numDigits = 0;
        for (auto ch : timestepPart) {
            if (ch == '_') {
                continue;
            }
            if (!isDigit(ch) || ++numDigits > MaxNumTimestepDigits) {
                return std::nullopt;
            }
            result = result * 10 + static_cast<uint64_t>(ch - '0');
        }
        return numDigits > 0 ? std::make_optional(result) : std::nullopt;
    }

    // The simulation that ALIEN saves when it is closed and loads when it is started
    bool isNamedLikeExitSave(std::string const& name)
    {
        return toAsciiUpper(name) == toAsciiUpper(StringHelper::toUtf8(Const::AutosaveFileWithoutPath.stem()));
    }

    bool isNamedLikeAutosave(std::string const& name)
    {
        return parseTimestep(name, Const::AutosavePrefix).has_value() || parseTimestep(name, Const::SavePointPrefix).has_value() || isNamedLikeExitSave(name);
    }

    std::string formatFileTime(std::filesystem::file_time_type const& fileTime)
    {
        // The clocks are related through their current times, which needs no time zone database
        auto systemTime = std::chrono::system_clock::now() + (fileTime - std::filesystem::file_time_type::clock::now());
        return StringHelper::format(std::chrono::time_point_cast<std::chrono::system_clock::duration>(systemTime));
    }

    std::string getName(std::filesystem::path const& folder)
    {
        return StringHelper::toUtf8(folder.has_filename() ? folder.filename() : folder);
    }

    std::filesystem::path normalizePath(std::filesystem::path const& folder)
    {
        std::error_code error;
        auto result = std::filesystem::weakly_canonical(folder, error);
        if (error) {
            result = std::filesystem::absolute(folder, error);
            if (error) {
                result = folder;
            }
        }
        result = result.lexically_normal();
        if (!result.has_filename() && result != result.root_path()) {
            result = result.parent_path();
        }
        return result;
    }

    // A file that belongs to a simulation carries its name. One that belongs to a genome carries its whole file name.
    std::filesystem::path getCompanionPath(std::filesystem::path const& path, std::string const& extension)
    {
        auto result = path;
        if (hasExtension(path, GenomeExtension)) {
            result += extension;
        } else {
            result.replace_extension(extension);
        }
        return result;
    }

    // The picture and the description of a simulation or a genome
    std::vector<std::filesystem::path> getCompanionPaths(std::filesystem::path const& path)
    {
        return {getCompanionPath(path, PictureExtension), getCompanionPath(path, DescriptionExtension)};
    }

    std::filesystem::path getFilePath(std::filesystem::path const& libraryFolder, std::string const& name, std::string const& extension)
    {
        return (libraryFolder / StringHelper::toPath(name + extension)).make_preferred();
    }

    SimulationDescription readDescription(std::filesystem::path const& descriptionPath)
    {
        try {
            std::ifstream stream(descriptionPath, std::ios::binary);
            if (!stream) {
                return {};
            }
            SimulationDescription result;
            auto maxLength = static_cast<size_t>(Const::MaxSimulationDescriptionLength);
            auto& text = result.text;
            text.resize(maxLength + 1);
            stream.read(text.data(), static_cast<std::streamsize>(text.size()));
            text.resize(static_cast<size_t>(stream.gcount()));
            if (text.size() > maxLength) {
                auto length = maxLength;
                while (length > 0 && isUtf8ContinuationByte(text.at(length))) {
                    --length;
                }
                text.resize(length);
                result.complete = false;
            }
            if (text.starts_with(Utf8ByteOrderMark)) {
                text.erase(0, std::string_view(Utf8ByteOrderMark).size());
            }
            std::erase(text, '\r');
            for (auto& ch : text) {
                if (static_cast<unsigned char>(ch) < 32 && ch != '\n') {
                    ch = ' ';
                }
            }
            auto first = text.find_first_not_of(" \n");
            text = first == std::string::npos ? std::string() : text.substr(first, text.find_last_not_of(" \n") - first + 1);
            return result;
        } catch (...) {
            return {};
        }
    }

    struct ScanState
    {
        int numFolders = 0;
        bool truncated = false;
    };

    SimulationFile
    createLibraryFile(std::filesystem::directory_entry const& entry, std::string const& namePrefix, std::set<std::string> const& companionFilenames)
    {
        auto const& path = entry.path();
        SimulationFile result{.name = namePrefix + StringHelper::toUtf8(path.stem()), .path = path};

        std::error_code error;
        auto lastChange = entry.last_write_time(error);
        if (!error) {
            result.timestamp = formatFileTime(lastChange);
        }
        auto fileSize = entry.file_size(error);
        result.fileSize = error ? 0 : fileSize;

        auto hasCompanion = [&](std::string const& extension) {
            return companionFilenames.contains(toAsciiUpper(StringHelper::toUtf8(getCompanionPath(path, extension).filename())));
        };
        result.hasPicture = hasCompanion(PictureExtension);
        if (hasCompanion(DescriptionExtension)) {
            auto description = readDescription(getCompanionPath(path, DescriptionExtension));
            result.description = description.text;
            result.descriptionComplete = description.complete;
        }
        return result;
    }

    void scanFolder(SimulationLibraryContent& content, std::filesystem::path const& folder, std::string const& namePrefix, int depth, ScanState& state)
    {
        std::vector<std::filesystem::directory_entry> simulationEntries;
        std::vector<std::filesystem::directory_entry> genomeEntries;
        std::set<std::string> companionFilenames;
        std::vector<std::filesystem::path> subFolders;
        forEachEntry(folder, [&](std::filesystem::directory_entry const& entry) {
            if (isSimulationFile(entry)) {
                simulationEntries.emplace_back(entry);
            } else if (isGenomeFile(entry)) {
                genomeEntries.emplace_back(entry);
            } else if (isRealFile(entry)) {
                if (hasExtension(entry.path(), PictureExtension) || hasExtension(entry.path(), DescriptionExtension)) {
                    companionFilenames.insert(toAsciiUpper(StringHelper::toUtf8(entry.path().filename())));
                }
            } else if (isLibraryFolder(entry)) {
                subFolders.emplace_back(entry.path());
            }
        });

        for (auto const& entry : simulationEntries) {
            if (toInt(content.simulations.size()) >= MaxNumFilesOfOneType) {
                state.truncated = true;
                break;
            }
            auto file = createLibraryFile(entry, namePrefix, companionFilenames);
            auto stem = StringHelper::toUtf8(entry.path().stem());
            if (auto autosaveTimestep = parseTimestep(stem, Const::AutosavePrefix)) {
                file.kind = SimulationFileKind::Autosave;
                file.timestep = autosaveTimestep;
            } else if (auto savePointTimestep = parseTimestep(stem, Const::SavePointPrefix)) {
                file.kind = SimulationFileKind::SavePoint;
                file.timestep = savePointTimestep;
            } else if (isNamedLikeExitSave(stem)) {
                file.kind = SimulationFileKind::Autosave;
            }
            content.simulations.emplace_back(file);
        }
        for (auto const& entry : genomeEntries) {
            if (toInt(content.genomes.size()) >= MaxNumFilesOfOneType) {
                state.truncated = true;
                break;
            }
            content.genomes.emplace_back(createLibraryFile(entry, namePrefix, companionFilenames));
        }

        std::ranges::sort(subFolders, [](std::filesystem::path const& left, std::filesystem::path const& right) {
            return StringHelper::naturalLess(getName(left), getName(right));
        });
        for (auto const& subFolder : subFolders) {
            if (depth >= MaxFolderDepth || state.numFolders >= MaxNumFolders) {
                state.truncated = true;
                return;
            }
            ++state.numFolders;
            scanFolder(content, subFolder, namePrefix + getName(subFolder) + NameSeparator, depth + 1, state);
        }
    }

    bool containsLibraryFiles(std::filesystem::path const& folder)
    {
        auto result = false;
        forEachEntry(folder, [&](std::filesystem::directory_entry const& entry) { result = result || isSimulationFile(entry) || isGenomeFile(entry); });
        return result;
    }

    bool containsNothingButSavepointTable(std::filesystem::path const& folder)
    {
        auto numEntries = 0;
        auto containsTable = false;
        forEachEntry(folder, [&](std::filesystem::directory_entry const& entry) {
            ++numEntries;
            containsTable = containsTable || StringHelper::toUtf8(entry.path().filename()) == SavepointTableFilename;
        });
        return numEntries == 1 && containsTable;
    }

    // Removes the folder and its parent folders inside the library as far as they have become empty. The library folder itself is kept.
    void removeEmptyFolders(std::filesystem::path const& folder, std::filesystem::path const& libraryFolder)
    {
        auto normalizedLibraryFolder = normalizePath(libraryFolder);
        auto currentFolder = normalizePath(folder);
        while (true) {
            auto relativePath = currentFolder.lexically_relative(normalizedLibraryFolder);
            if (relativePath.empty() || relativePath == "." || *relativePath.begin() == ".." || containsLibraryFiles(currentFolder)) {
                return;
            }
            // The table of the save points is removed with its folder, but not from a folder that stays because of its other content
            std::error_code error;
            if (containsNothingButSavepointTable(currentFolder)) {
                std::filesystem::remove(currentFolder / SavepointTableFilename, error);
            }

            // Removes the folder only if it is empty
            if (!std::filesystem::remove(currentFolder, error) || error) {
                return;
            }
            currentFolder = currentFolder.parent_path();
        }
    }

    // Removes a simulation or a genome with its picture and its description
    void removeLibraryFiles(std::filesystem::path const& path)
    {
        std::error_code error;
        std::filesystem::remove(path, error);
        for (auto const& companionPath : getCompanionPaths(path)) {
            std::filesystem::remove(companionPath, error);
        }
    }

    // A simulation must be named neither like an autosave, whose files are deleted or replaced, nor like a genome file, whose picture and
    // description it would share
    bool isValidName(std::string const& name, int maxNumFolders, bool isSimulation)
    {
        auto parts = splitName(name);
        if (toInt(parts.size()) > maxNumFolders + 1 || !std::ranges::all_of(parts, isValidNamePart)) {
            return false;
        }
        return !isSimulation || (!isNamedLikeAutosave(parts.back()) && !hasExtension(StringHelper::toPath(parts.back()), GenomeExtension));
    }

    // The folders in front of the name of a simulation
    bool areValidFolders(std::vector<std::string> const& nameParts)
    {
        auto folders = nameParts | std::views::take(toInt(nameParts.size()) - 1);
        return toInt(nameParts.size()) <= MaxFolderDepth + 1 && std::ranges::all_of(folders, isValidNamePart);
    }
}

SimulationLibraryContent SimulationLibraryService::loadContent(std::filesystem::path const& libraryFolder) const
{
    SimulationLibraryContent result;
    ScanState state;
    scanFolder(result, normalize(libraryFolder), std::string(), 0, state);
    result.truncated = state.truncated;

    auto hasLowerName = [](SimulationFile const& left, SimulationFile const& right) { return StringHelper::naturalLess(left.name, right.name); };
    std::ranges::stable_sort(result.simulations, hasLowerName);
    std::ranges::stable_sort(result.genomes, hasLowerName);
    return result;
}

bool SimulationLibraryService::isValidSimulationName(std::string const& name) const
{
    return isValidName(name, MaxFolderDepth, true);
}

bool SimulationLibraryService::isValidFolderName(std::string const& name) const
{
    return isValidName(name, MaxFolderDepth - 1, false);
}

std::filesystem::path SimulationLibraryService::normalize(std::filesystem::path const& folder) const
{
    return normalizePath(folder);
}

bool SimulationLibraryService::isInside(std::filesystem::path const& folder, std::filesystem::path const& libraryFolder) const
{
    auto relativePath = normalize(folder).lexically_relative(normalize(libraryFolder));
    return !relativePath.empty() && *relativePath.begin() != "..";
}

std::filesystem::path SimulationLibraryService::getSimulationPath(std::filesystem::path const& libraryFolder, std::string const& name) const
{
    return getFilePath(libraryFolder, name, SimulationExtension);
}

std::optional<std::string> SimulationLibraryService::getSimulationName(std::filesystem::path const& libraryFolder, std::filesystem::path const& simulationPath)
    const
{
    auto folderName = getFolderName(libraryFolder, simulationPath.parent_path());
    if (!folderName.has_value()) {
        return std::nullopt;
    }
    auto name = StringHelper::toUtf8(simulationPath.stem());
    return folderName->empty() ? name : *folderName + NameSeparator + name;
}

std::optional<std::string> SimulationLibraryService::getFolderName(std::filesystem::path const& libraryFolder, std::filesystem::path const& folder) const
{
    auto relativeFolder = normalize(folder).lexically_relative(normalize(libraryFolder));
    if (relativeFolder.empty() || *relativeFolder.begin() == "..") {
        return std::nullopt;
    }
    std::string result;
    if (relativeFolder != ".") {
        for (auto const& part : relativeFolder) {
            result += (result.empty() ? "" : std::string(1, NameSeparator)) + StringHelper::toUtf8(part);
        }
    }
    return result;
}

bool SimulationLibraryService::isSameFolder(std::filesystem::path const& libraryFolder, std::string const& folderName, std::string const& otherFolderName) const
{
    // Other names of a folder, like a short name on Windows, are no spellings of its name
    auto parts = splitName(folderName);
    auto otherParts = splitName(otherFolderName);
    if (parts.size() != otherParts.size()) {
        return false;
    }
    for (auto const& [part, otherPart] : std::views::zip(parts, otherParts)) {
        if (toAsciiUpper(part) != toAsciiUpper(otherPart)) {
            return false;
        }
    }
    std::error_code error;
    return std::filesystem::equivalent(libraryFolder / StringHelper::toPath(folderName), libraryFolder / StringHelper::toPath(otherFolderName), error)
        && !error;
}

bool SimulationLibraryService::changeFolderSpelling(std::filesystem::path const& libraryFolder, std::string const& folderName, std::string const& newFolderName)
    const
{
    if (!isValidFolderName(newFolderName) || !isSameFolder(libraryFolder, folderName, newFolderName)) {
        return false;
    }
    auto parts = splitName(folderName);
    auto newParts = splitName(newFolderName);
    auto parentFolder = libraryFolder;
    for (auto const& [part, newPart] : std::views::zip(parts, newParts)) {
        auto newFolder = parentFolder / StringHelper::toPath(newPart);
        if (part != newPart) {
            std::error_code error;
            std::filesystem::rename(parentFolder / StringHelper::toPath(part), newFolder, error);
            if (error) {
                return false;
            }
        }
        parentFolder = newFolder;
    }
    return true;
}

std::filesystem::path SimulationLibraryService::getPicturePath(std::filesystem::path const& simulationPath) const
{
    return getCompanionPath(simulationPath, PictureExtension);
}

std::filesystem::path SimulationLibraryService::getDescriptionPath(std::filesystem::path const& simulationPath) const
{
    return getCompanionPath(simulationPath, DescriptionExtension);
}

bool SimulationLibraryService::isSimulationNameTaken(
    std::filesystem::path const& libraryFolder,
    std::string const& name,
    std::optional<std::filesystem::path> const& exceptSimulationPath) const
{
    return isNameTaken(libraryFolder, name, SimulationExtension, exceptSimulationPath);
}

std::optional<std::filesystem::path> SimulationLibraryService::moveSimulation(
    std::filesystem::path const& libraryFolder,
    std::filesystem::path const& simulationPath,
    std::string const& newName) const
{
    return moveFile(libraryFolder, simulationPath, newName, SimulationExtension, true);
}

void SimulationLibraryService::deleteSimulation(std::filesystem::path const& libraryFolder, std::filesystem::path const& simulationPath) const
{
    removeLibraryFiles(simulationPath);
    removeEmptyFolders(simulationPath.parent_path(), libraryFolder);
}

bool SimulationLibraryService::isValidGenomeName(std::string const& name) const
{
    return isValidName(name, MaxFolderDepth, false);
}

bool SimulationLibraryService::isGenomePath(std::filesystem::path const& path) const
{
    return hasExtension(path, GenomeExtension);
}

std::filesystem::path SimulationLibraryService::getGenomePath(std::filesystem::path const& libraryFolder, std::string const& name) const
{
    return getFilePath(libraryFolder, name, GenomeExtension);
}

bool SimulationLibraryService::isGenomeNameTaken(
    std::filesystem::path const& libraryFolder,
    std::string const& name,
    std::optional<std::filesystem::path> const& exceptGenomePath) const
{
    return isNameTaken(libraryFolder, name, GenomeExtension, exceptGenomePath);
}

std::optional<std::filesystem::path>
SimulationLibraryService::moveGenome(std::filesystem::path const& libraryFolder, std::filesystem::path const& genomePath, std::string const& newName) const
{
    return moveFile(libraryFolder, genomePath, newName, GenomeExtension, false);
}

void SimulationLibraryService::deleteGenome(std::filesystem::path const& libraryFolder, std::filesystem::path const& genomePath) const
{
    removeLibraryFiles(genomePath);
    removeEmptyFolders(genomePath.parent_path(), libraryFolder);
}

bool SimulationLibraryService::isNameTaken(
    std::filesystem::path const& libraryFolder,
    std::string const& name,
    std::string const& extension,
    std::optional<std::filesystem::path> const& exceptPath) const
{
    auto isTaken = [&](std::filesystem::path const& path, std::optional<std::filesystem::path> const& exceptedPath) {
        std::error_code error;
        if (!std::filesystem::exists(path, error)) {
            return false;
        }
        return !exceptedPath.has_value() || !std::filesystem::equivalent(path, *exceptedPath, error);
    };

    auto path = getFilePath(libraryFolder, name, extension);
    if (isTaken(path, exceptPath)) {
        return true;
    }
    auto companionPaths = getCompanionPaths(path);
    if (!exceptPath.has_value()) {
        return std::ranges::any_of(companionPaths, [&](std::filesystem::path const& companionPath) { return isTaken(companionPath, std::nullopt); });
    }
    auto exceptCompanionPaths = getCompanionPaths(*exceptPath);
    for (auto const& [companionPath, exceptCompanionPath] : std::views::zip(companionPaths, exceptCompanionPaths)) {
        if (isTaken(companionPath, exceptCompanionPath)) {
            return true;
        }
    }
    return false;
}

std::optional<std::filesystem::path> SimulationLibraryService::moveFile(
    std::filesystem::path const& libraryFolder,
    std::filesystem::path const& simulationPath,
    std::string const& newName,
    std::string const& extension,
    bool isSimulation) const
{
    // A simulation that keeps its own name is only moved, for instance if its folder is renamed. Its name is not judged then, so that an
    // autosave keeps its kind and a file with an unusual name stays with the other files of its folder.
    auto newNameParts = splitName(newName);
    auto keepsOwnName = newNameParts.back() == StringHelper::toUtf8(simulationPath.stem());
    auto isValid = keepsOwnName ? areValidFolders(newNameParts) : isValidName(newName, MaxFolderDepth, isSimulation);

    // A picture or a description with the new name that belongs to no simulation counts as taken, because it would have to be replaced
    if (!isValid || isNameTaken(libraryFolder, newName, extension, simulationPath)) {
        return std::nullopt;
    }

    // A folder name that only differs in its spelling, for instance in upper and lower case on Windows, names the folder the simulation is
    // already in. Moving the simulation would change nothing then, so the folder itself gets the new spelling.
    auto folderName = getFolderName(libraryFolder, simulationPath.parent_path());
    auto separatorPos = newName.rfind(NameSeparator);
    auto newFolderName = separatorPos != std::string::npos ? newName.substr(0, separatorPos) : std::string();
    if (folderName.has_value() && *folderName != newFolderName && isSameFolder(libraryFolder, *folderName, newFolderName)
        && !changeFolderSpelling(libraryFolder, *folderName, newFolderName)) {
        return std::nullopt;
    }
    auto result = getFilePath(libraryFolder, newName, extension);
    std::error_code error;
    std::filesystem::create_directories(result.parent_path(), error);
    std::filesystem::rename(simulationPath, result, error);
    if (error) {
        removeEmptyFolders(result.parent_path(), libraryFolder);
        return std::nullopt;
    }
    auto companionPaths = getCompanionPaths(simulationPath);
    auto newCompanionPaths = getCompanionPaths(result);
    for (auto const& [companionPath, newCompanionPath] : std::views::zip(companionPaths, newCompanionPaths)) {
        if (std::filesystem::is_regular_file(companionPath, error)) {
            std::filesystem::rename(companionPath, newCompanionPath, error);
        }
    }
    removeEmptyFolders(simulationPath.parent_path(), libraryFolder);
    return result;
}

bool SimulationLibraryService::savePicture(std::filesystem::path const& simulationPath, std::string const& jpg) const
{
    try {
        std::ofstream stream(getPicturePath(simulationPath), std::ios::binary);
        stream.write(jpg.data(), static_cast<std::streamsize>(jpg.size()));
        return static_cast<bool>(stream);
    } catch (...) {
        return false;
    }
}

std::optional<std::string> SimulationLibraryService::loadPicture(std::filesystem::path const& simulationPath) const
{
    try {
        std::ifstream stream(getPicturePath(simulationPath), std::ios::binary);
        if (!stream) {
            return std::nullopt;
        }
        return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    } catch (...) {
        return std::nullopt;
    }
}

bool SimulationLibraryService::saveDescription(std::filesystem::path const& simulationPath, std::string const& description) const
{
    try {
        auto descriptionPath = getDescriptionPath(simulationPath);
        std::error_code error;
        if (description.empty()) {
            std::filesystem::remove(descriptionPath, error);
            return !error;
        }

        // Writing a temporary file first keeps the previous file intact if writing fails
        auto temporaryPath = descriptionPath;
        temporaryPath += TemporaryExtension;
        {
            std::ofstream stream(temporaryPath, std::ios::binary);
            stream.write(description.data(), static_cast<std::streamsize>(description.size()));
            if (!stream) {
                return false;
            }
        }
        std::filesystem::rename(temporaryPath, descriptionPath, error);
        return !error;
    } catch (...) {
        return false;
    }
}

SimulationDescription SimulationLibraryService::loadDescription(std::filesystem::path const& simulationPath) const
{
    return readDescription(getDescriptionPath(simulationPath));
}

void SimulationLibraryService::removeExcessAutosaves(std::filesystem::path const& folder, int maxNumAutosaves) const
{
    struct Autosave
    {
        std::filesystem::path path;
        std::filesystem::file_time_type lastChange;
    };
    std::vector<Autosave> autosaves;
    forEachEntry(folder, [&](std::filesystem::directory_entry const& entry) {
        if (!isSimulationFile(entry) || !parseTimestep(StringHelper::toUtf8(entry.path().stem()), Const::AutosavePrefix).has_value()) {
            return;
        }
        std::error_code error;
        autosaves.emplace_back(Autosave{.path = entry.path(), .lastChange = entry.last_write_time(error)});
    });
    std::ranges::sort(autosaves, {}, &Autosave::lastChange);

    auto numAutosavesToRemove = std::max(0, toInt(autosaves.size()) - std::max(0, maxNumAutosaves));
    for (auto const& autosave : autosaves | std::views::take(numAutosavesToRemove)) {
        removeLibraryFiles(autosave.path);
    }
}
