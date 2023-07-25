#pragma once

namespace Const
{
    std::string const ProgramVersion = "4.0.0.beta.27";

    std::string const BasePath = "resources/";

    auto const LogFilename = "log.txt";
    auto const AutosaveFileWithoutPath = "autosave.sim";
    auto const AutosaveFile = BasePath + AutosaveFileWithoutPath;
    auto const SettingsFilename = BasePath + "settings.json";

    auto const SimulationFragmentShader = BasePath + "shader.fs";
    auto const SimulationVertexShader = BasePath + "shader.vs";

    auto const EditorOnFilename = BasePath + "editor on.png";
    auto const EditorOffFilename = BasePath + "editor off.png";

    auto const RunFilename = BasePath + "run.png";
    auto const PauseFilename = BasePath + "pause.png";
    auto const StepBackwardFilename = BasePath + "step backward.png";
    auto const StepForwardFilename = BasePath + "step forward.png";
    auto const SnapshotFilename = BasePath + "snapshot.png";
    auto const RestoreFilname = BasePath + "restore.png";

    auto const ZoomInFilename = BasePath + "zoom in.png";
    auto const ZoomOutFilename = BasePath + "zoom out.png";
    auto const ResizeFilename = BasePath + "resize.png";

    auto const LogoFilename = BasePath + "logo.png";
}
