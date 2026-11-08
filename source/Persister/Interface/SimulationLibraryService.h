#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <Base/Singleton.h>

#include "SimulationLibrary.h"

// The simulation library is a folder with nested folders, which contain simulation files. A simulation is named by its path inside the
// library. A picture and a text file with the name of a simulation file hold its preview and its description.
class SimulationLibraryService
{
    MAKE_SINGLETON(SimulationLibraryService);

public:
    SimulationLibraryContent loadContent(std::filesystem::path const& libraryFolder) const;

    bool isValidSimulationName(std::string const& name) const;  // Names that autosave uses are refused, because their files are deleted or replaced
    bool isValidFolderName(std::string const& name) const;
    std::filesystem::path normalize(std::filesystem::path const& folder) const;                            // Absolute and without a trailing separator
    bool isInside(std::filesystem::path const& folder, std::filesystem::path const& libraryFolder) const;  // Also true for the library folder itself

    std::filesystem::path getSimulationPath(std::filesystem::path const& libraryFolder, std::string const& name) const;
    std::optional<std::string> getSimulationName(std::filesystem::path const& libraryFolder, std::filesystem::path const& simulationPath) const;
    std::optional<std::string> getFolderName(std::filesystem::path const& libraryFolder, std::filesystem::path const& folder) const;
    std::filesystem::path getPicturePath(std::filesystem::path const& simulationPath) const;
    std::filesystem::path getDescriptionPath(std::filesystem::path const& simulationPath) const;

    // Names that differ only in upper and lower case belong to the same folder on file systems that do not tell them apart, like on Windows.
    // Such a folder is renamed itself in order to change the spelling of its name. Moving a simulation does so if its new name spells the
    // folder it is in differently.
    bool isSameFolder(std::filesystem::path const& libraryFolder, std::string const& folderName, std::string const& otherFolderName) const;
    bool changeFolderSpelling(std::filesystem::path const& libraryFolder, std::string const& folderName, std::string const& newFolderName) const;

    // True if a simulation, a picture or a description with this name exists, apart from the given simulation and what belongs to it
    bool isSimulationNameTaken(
        std::filesystem::path const& libraryFolder,
        std::string const& name,
        std::optional<std::filesystem::path> const& exceptSimulationPath = std::nullopt) const;

    // A simulation is moved and deleted together with its picture and its description. Folders that become empty are removed.
    std::optional<std::filesystem::path>
    moveSimulation(std::filesystem::path const& libraryFolder, std::filesystem::path const& simulationPath, std::string const& newName) const;
    void deleteSimulation(std::filesystem::path const& libraryFolder, std::filesystem::path const& simulationPath) const;

    bool savePicture(std::filesystem::path const& simulationPath, std::string const& jpg) const;
    std::optional<std::string> loadPicture(std::filesystem::path const& simulationPath) const;
    bool saveDescription(std::filesystem::path const& simulationPath, std::string const& description) const;
    SimulationDescription loadDescription(std::filesystem::path const& simulationPath) const;

    // The library holds genome files as well. A genome is named, moved and deleted like a simulation, except that every valid name is free
    // for it. Its picture and its description carry its whole file name, so that a genome and a simulation can have the same name. The
    // functions for pictures and descriptions take the path of a genome as well.
    bool isValidGenomeName(std::string const& name) const;
    bool isGenomePath(std::filesystem::path const& path) const;
    std::filesystem::path getGenomePath(std::filesystem::path const& libraryFolder, std::string const& name) const;
    bool isGenomeNameTaken(
        std::filesystem::path const& libraryFolder,
        std::string const& name,
        std::optional<std::filesystem::path> const& exceptGenomePath = std::nullopt) const;
    std::optional<std::filesystem::path>
    moveGenome(std::filesystem::path const& libraryFolder, std::filesystem::path const& genomePath, std::string const& newName) const;
    void deleteGenome(std::filesystem::path const& libraryFolder, std::filesystem::path const& genomePath) const;

    // Deletes the oldest simulations that autosave has created in the folder beyond the given number
    void removeExcessAutosaves(std::filesystem::path const& folder, int maxNumAutosaves) const;

private:
    bool isNameTaken(
        std::filesystem::path const& libraryFolder,
        std::string const& name,
        std::string const& extension,
        std::optional<std::filesystem::path> const& exceptPath) const;
    std::optional<std::filesystem::path> moveFile(
        std::filesystem::path const& libraryFolder,
        std::filesystem::path const& path,
        std::string const& newName,
        std::string const& extension,
        bool isSimulation) const;
};
