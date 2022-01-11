#pragma once

namespace Const
{
    std::string const ProgramVersion = "3 preview";

    std::string const BasePath = "resources/";

    auto const LogFilename = "log.txt";
    auto const AutosaveFile = BasePath + "autosave.sim";
    auto const SettingsFilename = BasePath + "settings.json";

    auto const SimulationFragmentShader = BasePath + "shader.fs";
    auto const SimulationVertexShader = BasePath + "shader.vs";

    auto const NavigationOnFilename = BasePath + "navigation on.png";
    auto const NavigationOffFilename = BasePath + "navigation off.png";
    auto const ActionOnFilename = BasePath + "action on.png";
    auto const ActionOffFilename = BasePath + "action off.png";

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
