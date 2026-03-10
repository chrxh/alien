#pragma once

#include <filesystem>

namespace Const
{
    std::string const ProgramVersion = "5.0.0-alpha.10";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const AlienURL = "alien-project.org";

    std::filesystem::path const ResourcePath = "resources";
    std::filesystem::path const AutosavePath = ResourcePath / "autosave";
    std::filesystem::path const ImagesPath = ResourcePath / "images";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = AutosavePath / AutosaveFileWithoutPath;
    std::filesystem::path const SettingsFilename = AutosavePath / "settings.json";
    std::filesystem::path const SavepointTableFilename = "savepoints.json";

    std::filesystem::path const EditorOnFilename = ImagesPath / "editor on.png";
    std::filesystem::path const EditorOffFilename = ImagesPath / "editor off.png";

    std::filesystem::path const LogoFilename = ImagesPath / "logo.png";
}
