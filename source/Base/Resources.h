#pragma once

#include <filesystem>

namespace Const
{
    std::string const ProgramVersion = "5.0.0-alpha.0";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const AlienURL = "alien-project.org";

    std::filesystem::path const ResourcePath = "resources";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = ResourcePath / AutosaveFileWithoutPath;
    std::filesystem::path const SettingsFilename = ResourcePath / "settings.json";
    std::filesystem::path const SavepointTableFilename = "savepoints.json";

    std::filesystem::path const SimulationFragmentShader = ResourcePath / "shader.fs";
    std::filesystem::path const SimulationVertexShader = ResourcePath / "shader.vs";

    std::filesystem::path const EditorOnFilename = ResourcePath / "editor on.png";
    std::filesystem::path const EditorOffFilename = ResourcePath / "editor off.png";

    std::filesystem::path const LogoFilename = ResourcePath / "logo.png";
}
