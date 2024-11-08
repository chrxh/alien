#pragma once

#include <filesystem>

namespace Const
{
    std::string const ProgramVersion = "4.11.0";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const AlienURL = "alien-project.org";

    std::filesystem::path const BasePath = "resources";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = BasePath / AutosaveFileWithoutPath;
    std::filesystem::path const SettingsFilename = BasePath / "settings.json";
    std::filesystem::path const SavepointTableFilename = "savepoints.json";

    std::filesystem::path const SimulationFragmentShader = BasePath / "shader.fs";
    std::filesystem::path const SimulationVertexShader = BasePath / "shader.vs";

    std::filesystem::path const EditorOnFilename = BasePath / "editor on.png";
    std::filesystem::path const EditorOffFilename = BasePath / "editor off.png";

    std::filesystem::path const LogoFilename = BasePath / "logo.png";
}
