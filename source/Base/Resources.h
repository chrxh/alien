#pragma once

#include <filesystem>
#include <string>

namespace Const
{
    extern std::string const ProgramVersion;
    extern std::string const DiscordURL;
    extern std::string const AlienServerURL;
    extern std::string const StartupSimulationResourceName;

    extern std::filesystem::path const ResourcePath;
    extern std::filesystem::path const AutosavePath;
    extern std::filesystem::path const ImagesPath;

    extern std::filesystem::path const LogFilename;
    extern std::filesystem::path const ProfileFilename;
    extern std::filesystem::path const TraceFilename;
    extern std::filesystem::path const AutosaveFileWithoutPath;
    extern std::filesystem::path const AutosaveFile;
    extern std::filesystem::path const SettingsFilename;
    extern std::filesystem::path const SavepointTableFilename;

    extern std::filesystem::path const LogoFilename;
}
