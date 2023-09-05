#pragma once

namespace Const
{
    std::string const ProgramVersion = "4.0.3";

    std::string const BasePath = "resources/";

    auto const LogFilename = "log.txt";
    auto const AutosaveFileWithoutPath = "autosave.sim";
    auto const AutosaveFile = BasePath + AutosaveFileWithoutPath;
    auto const SettingsFilename = BasePath + "settings.json";

    auto const SimulationFragmentShader = BasePath + "shader.fs";
    auto const SimulationVertexShader = BasePath + "shader.vs";

    auto const EditorOnFilename = BasePath + "editor on.png";
    auto const EditorOffFilename = BasePath + "editor off.png";

    auto const LogoFilename = BasePath + "logo.png";
}
