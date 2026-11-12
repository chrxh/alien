#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace Const
{
    extern std::string const ProgramVersion;
    extern std::string const DiscordURL;
    extern std::string const PrivacyPolicyURL;
    extern std::string const AlienServerURL;
    extern std::string const StartupSimulationResourceName;

    extern std::filesystem::path const ResourcePath;
    extern std::filesystem::path const AutosavePath;
    extern std::filesystem::path const ImagesPath;
    extern std::filesystem::path const DocsPath;

    extern std::filesystem::path const LogFilename;
    extern std::filesystem::path const ProfileFilename;
    extern std::filesystem::path const TraceFilename;
    extern std::filesystem::path const AutosaveFileWithoutPath;
    extern std::filesystem::path const AutosaveFile;
    extern std::filesystem::path const SavepointTableFilename;

    extern std::filesystem::path const LogoCoreFilename;
    extern std::filesystem::path const LogoGlowFilename;
    extern std::vector<std::filesystem::path> const WindowIconFilenames;
    extern std::filesystem::path const CaBundleFilename;
}
