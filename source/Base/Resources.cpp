#include "Resources.h"

#ifdef _WIN32
#include <Windows.h>
#endif

namespace
{
    std::filesystem::path getProgramDirectory()
    {
#ifdef _WIN32
        auto constexpr MaxPathLength = 32768;
        std::wstring programFilename(MaxPathLength, L'\0');
        auto length = GetModuleFileNameW(nullptr, programFilename.data(), MaxPathLength);
        if (length == 0) {
            return {};
        }
        programFilename.resize(length);
        return std::filesystem::path(programFilename).parent_path();
#else
        std::error_code errorCode;
        auto programFilename = std::filesystem::read_symlink("/proc/self/exe", errorCode);
        return errorCode ? std::filesystem::path() : programFilename.parent_path();
#endif
    }
}

namespace Const
{
    std::string const ProgramVersion = "5.0.0-beta.2";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const PrivacyPolicyURL = "https://alien-project.org/privacy.html";
    std::string const AlienServerURL = "api.alien-project.org";
    std::string const StartupSimulationResourceName = "Startup/Version1";

    std::filesystem::path const ResourcePath = "resources";
    std::filesystem::path const AutosavePath = ResourcePath / "autosave";
    std::filesystem::path const SimulationLibraryPath = "simulations";
    std::filesystem::path const ImagesPath = ResourcePath / "images";
    std::filesystem::path const DocsPath = ResourcePath / "docs";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const ProfileFilename = "profile.txt";
    std::filesystem::path const TraceFilename = "trace.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = AutosavePath / AutosaveFileWithoutPath;

    std::filesystem::path const LogoCoreFilename = ImagesPath / "logoCore.png";
    std::filesystem::path const LogoGlowFilename = ImagesPath / "logoGlow.png";
    std::vector<std::filesystem::path> const WindowIconFilenames =
        {ImagesPath / "alien16.png", ImagesPath / "alien32.png", ImagesPath / "alien48.png", ImagesPath / "alien256.png"};
    std::filesystem::path const CaBundleFilename = getProgramDirectory() / ResourcePath / "ca-bundle.crt";
}
