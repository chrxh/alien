#pragma once

namespace Const
{
    std::string const ProgramVersion = "4.4.3";
    std::string const DiscordLink = "https://discord.gg/7bjyZdXXQ2";

    std::string const BasePath = "resources/";

    std::string const LogFilename = "log.txt";
    std::string const AutosaveFileWithoutPath = "autosave.sim";
    std::string const AutosaveFile = BasePath + AutosaveFileWithoutPath;
    std::string const SettingsFilename = BasePath + "settings.json";

    std::string const SimulationFragmentShader = BasePath + "shader.fs";
    std::string const SimulationVertexShader = BasePath + "shader.vs";

    std::string const EditorOnFilename = BasePath + "editor on.png";
    std::string const EditorOffFilename = BasePath + "editor off.png";

    std::string const LogoFilename = BasePath + "logo.png";
}
