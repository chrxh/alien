#pragma once

#include <filesystem>
#include <optional>

struct ReadLocalLibraryRequestData
{
    std::filesystem::path libraryFolder;
    std::optional<std::filesystem::path> autosaveMainFolder;  // A folder outside the library whose autosaves are added
};
