#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace Const
{
    // Autosaves and the save points of earlier versions are named by their prefix followed by their time step
    auto constexpr SavePointPrefix = "save_";
    auto constexpr AutosavePrefix = "autosave_";

    auto constexpr MaxFileDescriptionLength = 4000;
}

enum class LibraryFileKind
{
    Named,
    SavePoint,  // Written by the Autosave window of earlier versions
    Autosave    // Written by autosave, or the simulation that ALIEN saves when it is closed
};

struct LibraryFile
{
    std::string name;  // Path inside the library without the extension and with slashes between folders, e.g. Hanging Garden/After 13 hours
    std::filesystem::path path;
    LibraryFileKind kind = LibraryFileKind::Named;
    std::string timestamp;  // Time of the last change
    uint64_t fileSize = 0;
    std::optional<uint64_t> timestep;  // Known from the names of autosaves and save points
    bool hasPicture = false;
    std::string description;
    bool descriptionComplete = true;  // False if the stored description is longer than the text, in which case it must not be replaced by the text
};

// The description of a simulation as far as it fits into the limit of its length
struct FileDescription
{
    std::string text;
    bool complete = true;
};

struct LocalLibraryContent
{
    std::vector<LibraryFile> simulations;  // Sorted by name
    std::vector<LibraryFile> genomes;      // Sorted by name, all of the kind Named
    bool truncated = false;                // A limit for the number of folders, their depth or the number of files has been reached
};
